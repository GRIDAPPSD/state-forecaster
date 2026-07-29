#!/usr/bin/python3
import sys
import os
import io
import time
import queue
import math
import logging
from collections import deque
from itertools import islice
from datetime import datetime, timezone
import json
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =====================================================
# CONFIG
# =====================================================

FORECAST_OUTPUT_JSONL = "forecast_output.jsonl"

#JSON_PATH = "results_data_forecasting_13_1min.jsonl"
JSON_PATH = "results_data_forecasting_13_5min.jsonl"
#JSON_PATH = "results_data_forecasting_13_15min.jsonl"
#JSON_PATH = "results_data_forecasting_123_5min.jsonl"

# --- streaming cadence ---
#TS_INCREMENT_SEC = 3            # timestamp spacing in the input stream (real-time)
#TS_INCREMENT_SEC = 60           # timestamp spacing in the input stream (1 min)
TS_INCREMENT_SEC = 300          # timestamp spacing in the input stream (5 min)
#TS_INCREMENT_SEC = 900          # timestamp spacing in the input stream (15 min)

COMPUTE_LIVE_MAE = True   # deferred scoring: score one forecast per model version
                          # against the actual estimates that later arrive. Off for
                          # production-speed runs.

CADENCE_CHECK_SAMPLES = 20      # real records to sample for the startup cadence check

# --- history / horizon (in SAMPLES, i.e. timestamps) ---
HIST = 15                      # past samples used as input (15 min @ 1-min)
FUT = 15                       # future samples to forecast (15 min @ 1-min)

# --- lag features (in TIME, converted to seconds) ---
DAY_LAG_SEC = 1 * 24 * 3600    # 1-day lag
WEEK_LAG_SEC = 7 * 24 * 3600   # 1-week lag

USE_CURRENT_PQ = True

# --- block / retention geometry ---
BLOCK_DAYS = 2                 # training block size ("2-day window")
RETENTION_DAYS = 10            # rolling buffer horizon (see rationale below)
# Rationale for RETENTION_DAYS:
#   * >= 7 days: distribution load has a WEEKLY trend; a full week must be
#     retained so the 1-week lag feature is populated (not zero-flagged).
#   * The current block's forecast references samples as far back as its START,
#     which needs 7 days *before* the block start -> ~9 days from block end.
#   * Rounded up to 10 so retention is an even multiple of BLOCK_DAYS (5 blocks)
#     and the week-lag never degrades at block boundaries.
#   All heuristics; kept configurable for later evaluation.

# --- training ---
EPOCHS_PER_BLOCK = 8
BATCH_SIZE = 512
NUM_WORKERS = 0
USE_GPU = True
DROPOUT_P = 0.03               # training regularization only (no MC dropout)
MAX_WINDOW_SAMPLES = 500_000   # cap: bounds per-block memory AND train time.
VAL_FRACTION = 0.05

# --- forecasting policy ---
# First block is TRAIN-ONLY (model must see >=1 block before forecasting).
# From block 2 onward: forecast the new block with the current model
# (true out-of-sample), THEN train on it. This forecast-then-train ordering
# is single-process scaffolding; it dissolves once train/forecast are split
# into separate processes sharing the model.
MIN_BLOCKS_BEFORE_FORECAST = 1

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() and USE_GPU else "cpu"
PIN_MEMORY = (device == "cuda")

# =====================================================
# DERIVED CONSTANTS (computed from config, nothing hardcoded)
# =====================================================
BLOCK_SEC = BLOCK_DAYS * 24 * 3600
RETENTION_SEC = RETENTION_DAYS * 24 * 3600
assert RETENTION_SEC >= WEEK_LAG_SEC + HIST * TS_INCREMENT_SEC, \
    "RETENTION too small: week-lag feature would be permanently disabled."

INPUT_DIM = (
    HIST * 2 +   # historical V, angle
    2 +          # current or previous P,Q
    2 +          # 1-day lag P,Q
    2 +          # 1-day lag V, angle
    1 +          # 1-day lag availability flag
    2 +          # 1-week lag P,Q
    2 +          # 1-week lag V, angle
    1 +          # 1-week lag availability flag
    3 +          # phase
    2            # sin_time, cos_time
)
OUTPUT_DIM = FUT * 2


#FEED_RATE_HZ = 1.0          # records/sec the feeder emits (real ~1; >1 speeds testing)
#FEED_RATE_HZ = 4.0          # records/sec the feeder emits (real ~1; >1 speeds testing)
FEED_RATE_HZ = 50.0          # records/sec the feeder emits (real ~1; >1 speeds testing)
FORECASTER_POLL_SEC = 0.05   # forecaster idle poll interval when no work is pending
FEEDER_POLL_SEC = 0.05       # feeder idle poll interval
LOG_DIR = "."
TRAINER_LOG = f"{LOG_DIR}/trainer.log"
FORECASTER_LOG = f"{LOG_DIR}/forecaster.log"
FEEDER_LOG = f"{LOG_DIR}/feeder.log"

DONE = "__DONE__"           # sentinel Queue item meaning "end of stream"


# =====================================================
# Start from forecaster_single.py
# =====================================================
def _pq_value(x):
    """Coerce a P or Q field to float. The real State Estimator publishes the
    string "NA" for buses without a P/Q estimate (e.g. SOURCEBUS); map those to
    0.0 (matches the prior simplified-file handling). None is also treated as 0.0."""
    if x == "NA" or x is None:
        return 0.0
    return float(x)


def parse_sv_entry(entry):
    """Parse one SvEstVoltages entry into (node_key, node_values).
    Single source of truth for entry interpretation, shared by the bus callback
    and the file reader. Node key = ConnectivityNode + "." + phase (Approach 3);
    phase used as-is, empty phase falls back to bare ConnectivityNode. vpu -> V
    (per-unit), ang (radians); P/Q "NA" -> 0.0. V/Angle not
    NA-coerced (missing voltage/angle should fail loudly)."""
    cn = entry["ConnectivityNode"]
    phase = entry["phase"]
    node_key = f"{cn}.{phase}" if phase != "" else cn
    node_vals = {
        "P": _pq_value(entry["P"]),
        "Q": _pq_value(entry["Q"]),
        "V": float(entry["vpu"]),
        "Angle": float(entry["angleRad"]),
    }
    return node_key, node_vals


def sv_entries_to_nodes(sv_entries):
    """Build the internal {node_key: {P,Q,V,Angle}} dict from a SvEstVoltages list."""
    nodes = {}
    for entry in sv_entries:
        node_key, node_vals = parse_sv_entry(entry)
        nodes[node_key] = node_vals
    return nodes


def read_json_records(path):
    """Yield one internal record per line from the SE-written .jsonl file.
    (Envelope: {"timeStamp": ..., "SvEstVoltages": [...]}.)"""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ts = int(rec["timeStamp"])
            nodes = sv_entries_to_nodes(rec["SvEstVoltages"])   # shared
            yield {"timestamp": ts, "nodes": nodes}


def _emit_cadence_warning(min_delta, increment_sec):
    """Loud, hard-to-miss warning when the detected data cadence doesn't match
    the configured TS_INCREMENT_SEC. Printed once. Goes to stdout (thech
    console in multi-process), so it surfaces prominently rather than being
    buried."""
    bar = "*" * 72
    print("\n" + bar)
    print("!!!!!!!!!!!!!!!!!!!!!!  CADENCE MISMATCH WARNING  !!!!!!!!!!!!!!!!!!!!!!")
    print(bar)
    print(f"***  CONFIGURED TS_INCREMENT_SEC         = {increment_sec} s")
    print(f"***  DETECTED data cadence (min spacing) = {min_delta} s")
    if min_delta > increment_sec:
        print("***  >>> CONFIG INCREMENT IS TOO SMALL FOR THIS DATA <<<")
        print("***  The imputer will FABRICATE interpolated records to fill")
        print("***  gaps that do not really exist, yielding results that look")
        print("***  PLAUSIBLE BUT ARE WRONG (and much slower training).")
    else:
        print("***  >>> CONFIG INCREMENT IS TOO LARGE FOR THIS DATA <<<")
        print("***  History positions and lag lookups will be MISALIGNED.")
    print(f"***  ACTION: set TS_INCREMENT_SEC = {min_delta} to match the data,")
    print("***  or supply state estimates at the configured cadence.")
    print(bar + "\n")


def make_imputer(increment_sec):
    """Create a stateful per-record imputation step for a gapless, grid-aligned
    stream. Returns a function `step(record) -> [records]`.

    Each call takes ONE internal-format record ({"timestamp", "nodes": {...}})
    and returns the list of records to emit for it: normally just [real], but
    when there's a gap since the previous real record, the linearly-interpolated
    placeholder(s) are prepended -> [imp, imp, ..., real] (a "burst"). Imputed
    records carry "_imputed": True; real records carry "_imputed": False.

    State (prev timestamp/nodes + cadence-check counters) is held in closure via
    nonlocal, so this works identically whether driven by a pull loop (file) or
    a push callback (GridAPPS-D bus). Also performs the one-time startup CADENCE
    CHECK on raw spacing of the first real records (min spacing = true cadence,
    since gaps only increase spacing); warns loudly once on mismatch.

    NO leading imputation: nothing is emitted before the first real record (it
    is "time zero"). Interpolation is per-node linear for P, Q, V, Angle.
    """
    prev_ts = None
    prev_nodes = None
    cad_min_delta = None
    cad_count = 0
    cad_warned = False
    cad_mismatch = False

    def step(record):
        nonlocal prev_ts, prev_nodes
        nonlocal cad_min_delta, cad_count, cad_warned, cad_mismatch

        ts = int(record["timestamp"])
        out = []

        if prev_ts is not None:
            delta = ts - prev_ts

            # --- cadence check on raw spacing (before imputation) ---
            if not cad_warned and delta > 0:
                cad_min_delta = delta if cad_min_delta is None else min(cad_min_delta, delta)
                cad_count += 1
                # conclude early on definitive evidence, else at the sample budget
                if cad_min_delta < increment_sec or cad_count >= CADENCE_CHECK_SAMPLES:
                    if cad_min_delta != increment_sec:
                        _emit_cadence_warning(cad_min_delta, increment_sec)
                        cad_mismatch = True
                    cad_warned = True

            if delta <= 0:
                print(f"[IMPUTE] WARNING: non-increasing timestamp "
                      f"{prev_ts} -> {ts}; passing through without imputation.")
            elif delta % increment_sec != 0:
                if not cad_mismatch:
                    print(f"[IMPUTE] WARNING: gap {delta}s not a multiple of "
                          f"increment {increment_sec}s ({prev_ts} -> {ts}); "
                          f"no imputation for this gap.")
            else:
                gap_steps = delta // increment_sec
                if gap_steps > 1:
                    for k in range(1, gap_steps):
                        frac = k / gap_steps
                        imp_ts = prev_ts + k * increment_sec
                        imp_nodes = {}
                        for node_name, cur_vals in record["nodes"].items():
                            pv = prev_nodes.get(node_name)
                            if pv is None:
                                imp_nodes[node_name] = dict(cur_vals)
                            else:
                                imp_nodes[node_name] = {
                                    "P": pv["P"] + frac * (cur_vals["P"] - pv["P"]),
                                    "Q": pv["Q"] + frac * (cur_vals["Q"] - pv["Q"]),
                                    "V": pv["V"] + frac * (cur_vals["V"] - pv["V"]),
                                    "Angle": pv["Angle"] + frac * (cur_vals["Angle"] - pv["Angle"]),
                                }
                        out.append({"timestamp": imp_ts, "nodes": imp_nodes, "_imputed": True})

        record["_imputed"] = False
        out.append(record)

        prev_ts = ts
        prev_nodes = record["nodes"]
        return out

    return step


# =====================================================
# TIME FEATURES (from epoch seconds; no pandas)
# =====================================================
def time_features(epoch_sec):
    """Return (sin_time, cos_time) for minute-of-day, matching the CSV pipeline
    which used UTC-naive timestamps derived from unix seconds."""
    dt = datetime.fromtimestamp(epoch_sec, tz=timezone.utc)
    minute_of_day = dt.hour * 60 + dt.minute
    ang = 2.0 * math.pi * minute_of_day / 1440.0
    return math.sin(ang), math.cos(ang)


def encode_phase(load_node):
    s = str(load_node).lower()
    p = s[-1]
    return {
        "a": np.array([1, 0, 0], dtype=np.float32),
        "b": np.array([0, 1, 0], dtype=np.float32),
        "c": np.array([0, 0, 1], dtype=np.float32),
        "1": np.array([1, 0, 0], dtype=np.float32),
        "2": np.array([0, 1, 0], dtype=np.float32),
        "3": np.array([0, 0, 1], dtype=np.float32),
    }.get(p, np.zeros(3, dtype=np.float32))


def assemble_input_vector(hist_va, base_pq, day_row, week_row, phase, time_feat):
    """Single source of truth for the model input layout.
    Used identically by the trainer (ReplayDataset) and the forecaster.

    Args (all torch.float32):
        hist_va  : [HIST*2] flattened V,ang of the HIST rows before base (position-based)
        base_pq  : [2] P,Q of the base row (t if USE_CURRENT_PQ else t-1)
        day_row  : [6] normalized row (V,ang,P,Q,sin,cos) at the 1-day lag, or None
        week_row : [6] normalized row at the 1-week lag, or None
        phase    : [3] phase one-hot
        time_feat: [2] sin_time, cos_time of the base row

    Returns X : [INPUT_DIM] float32
    """
    if day_row is None:
        day_pq = torch.zeros(2, dtype=torch.float32)
        day_va = torch.zeros(2, dtype=torch.float32)
        day_flag = torch.zeros(1, dtype=torch.float32)
    else:
        day_pq = day_row[2:4]
        day_va = day_row[0:2]
        day_flag = torch.ones(1, dtype=torch.float32)

    if week_row is None:
        week_pq = torch.zeros(2, dtype=torch.float32)
        week_va = torch.zeros(2, dtype=torch.float32)
        week_flag = torch.zeros(1, dtype=torch.float32)
    else:
        week_pq = week_row[2:4]
        week_va = week_row[0:2]
        week_flag = torch.ones(1, dtype=torch.float32)

    return torch.cat([
        hist_va,          # HIST * 2
        base_pq,          # 2
        day_pq,           # 2
        day_va,           # 2
        day_flag,         # 1
        week_pq,          # 2
        week_va,          # 2
        week_flag,        # 1
        phase,            # 3
        time_feat         # 2
    ])


# =====================================================
# INCREMENTAL SCALERS (replace sklearn MinMax/StandardScaler)
# Updated per-block with that block's raw values, causally.
# =====================================================
class RunningMinMax:
    """Streaming MinMaxScaler for a single feature. Maps to [0, 1]."""
    def __init__(self):
        self.min = math.inf
        self.max = -math.inf
    def update(self, values):
        if len(values) == 0:
            return
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if vmin < self.min:
            self.min = vmin
        if vmax > self.max:
            self.max = vmax
    def transform(self, x):
        rng = self.max - self.min
        if rng == 0 or not math.isfinite(rng):
            return np.zeros_like(x, dtype=np.float64)
        return (x - self.min) / rng
    def inverse(self, x_scaled):
        rng = self.max - self.min
        return x_scaled * rng + self.min


class RunningStandardizer:
    """Streaming StandardScaler via Welford/Chan parallel variance."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
    def update(self, values):
        if len(values) == 0:
            return
        b = np.asarray(values, dtype=np.float64)
        nb = b.size
        mb = float(b.mean())
        M2b = float(((b - mb) ** 2).sum())
        if self.n == 0:
            self.n, self.mean, self.M2 = nb, mb, M2b
            return
        delta = mb - self.mean
        tot = self.n + nb
        self.mean += delta * nb / tot
        self.M2 += M2b + delta * delta * self.n * nb / tot
        self.n = tot
    @property
    def std(self):
        if self.n < 2:
            return 1.0
        s = math.sqrt(self.M2 / self.n)
        return s if s > 0 else 1.0
    def transform(self, x):
        return (x - self.mean) / self.std
    def inverse(self, x_scaled):
        return x_scaled * self.std + self.mean


def extract_scaler_state(buf):
    """Snapshot the four incremental scalers' state (plain picklable numbers)."""
    return {
        "P":   (buf.sc_P.min, buf.sc_P.max),
        "Q":   (buf.sc_Q.min, buf.sc_Q.max),
        "V":   (buf.sc_V.min, buf.sc_V.max),
        "ang": (buf.sc_ang.n, buf.sc_ang.mean, buf.sc_ang.M2),
    }


def apply_scaler_state(buf, state):
    """Restore scaler state into a buffer's scalers (forecaster side)."""
    buf.sc_P.min, buf.sc_P.max = state["P"]
    buf.sc_Q.min, buf.sc_Q.max = state["Q"]
    buf.sc_V.min, buf.sc_V.max = state["V"]
    buf.sc_ang.n, buf.sc_ang.mean, buf.sc_ang.M2 = state["ang"]


# =====================================================
# ROLLING BUFFER
# Per-node raw ring of (ts, V, ang, P, Q). Normalized tensors + timestamp->pos
# lookups are rebuilt each block after scalers update. Old rows evicted past
# RETENTION_SEC measured from the newest timestamp.
# =====================================================
class RollingBuffer:
    def __init__(self, node_names):
        self.node_names = list(node_names)
        self.node_to_id = {n: i for i, n in enumerate(self.node_names)}
        self.id_to_node = {i: n for n, i in self.node_to_id.items()}
        self.num_nodes = len(self.node_names)
        # per node_id: deque of (ts, V, ang, P, Q) in time order
        self.raw = {nid: deque() for nid in range(self.num_nodes)}
        self.phase = {nid: torch.tensor(encode_phase(n), dtype=torch.float32)
                      for nid, n in self.id_to_node.items()}
        # rebuilt each block:
        self.tensors = {}       # nid -> float32 [T,6]: V,ang,P,Q,sin,cos (normalized)
        self.pos_by_ts = {}     # nid -> {ts: row index}
        self.newest_ts = None
        # scalers
        self.sc_P = RunningMinMax()
        self.sc_Q = RunningMinMax()
        self.sc_V = RunningMinMax()
        self.sc_ang = RunningStandardizer()

    def append_record(self, record):
        """Add one timestamp's worth of node values (raw)."""
        ts = int(record["timestamp"])
        self.newest_ts = ts
        for node_name, vals in record["nodes"].items():
            nid = self.node_to_id.get(node_name)
            if nid is None:
                continue  # node not seen in first record; fixed node set assumed
            P = vals["P"] if vals["P"] is not None else 0.0
            Q = vals["Q"] if vals["Q"] is not None else 0.0
            self.raw[nid].append((ts, vals["V"], vals["Angle"], P, Q))

    def evict_old(self):
        """Drop rows older than RETENTION_SEC behind the newest timestamp."""
        if self.newest_ts is None:
            return
        cutoff = self.newest_ts - RETENTION_SEC
        for nid, dq in self.raw.items():
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    def update_scalers_with_block(self, block_start, block_end):
        """Update running scalers using ONLY this block's raw values (causal)."""
        Ps, Qs, Vs, As = [], [], [], []
        for dq in self.raw.values():
            for (ts, V, A, P, Q) in dq:
                if block_start <= ts < block_end:
                    Vs.append(V); As.append(A); Ps.append(P); Qs.append(Q)
        self.sc_V.update(np.asarray(Vs, dtype=np.float64))
        self.sc_ang.update(np.asarray(As, dtype=np.float64))
        self.sc_P.update(np.asarray(Ps, dtype=np.float64))
        self.sc_Q.update(np.asarray(Qs, dtype=np.float64))

    def rebuild_normalized(self):
        """Rebuild per-node normalized tensors + ts->pos maps from raw buffers,
        using the CURRENT scaler state. Called once per block after scalers update."""
        self.tensors = {}
        self.pos_by_ts = {}
        for nid, dq in self.raw.items():
            if not dq:
                self.tensors[nid] = torch.empty((0, 6), dtype=torch.float32)
                self.pos_by_ts[nid] = {}
                continue
            arr = np.asarray(dq, dtype=np.float64)  # [T,5]: ts,V,ang,P,Q
            ts_col = arr[:, 0].astype(np.int64)
            Vn = self.sc_V.transform(arr[:, 1])
            An = self.sc_ang.transform(arr[:, 2])
            Pn = self.sc_P.transform(arr[:, 3])
            Qn = self.sc_Q.transform(arr[:, 4])
            sins = np.empty(len(arr), dtype=np.float64)
            coss = np.empty(len(arr), dtype=np.float64)
            for i, ts in enumerate(ts_col):
                s, c = time_features(int(ts))
                sins[i] = s; coss[i] = c
            mat = np.stack([Vn, An, Pn, Qn, sins, coss], axis=1)
            self.tensors[nid] = torch.tensor(mat, dtype=torch.float32)
            self.pos_by_ts[nid] = {int(ts): i for i, ts in enumerate(ts_col)}

    def _valid_positions(self, nid):
        """Positions with full HIST history behind and FUT targets ahead."""
        T = self.tensors[nid].shape[0]
        return range(HIST, T - FUT)

    def build_training_indices(self):
        """All valid samples in the buffer, subsampled to MAX_WINDOW_SAMPLES.
        Newest block's samples are always kept; older samples subsampled."""
        idx = []
        for nid in range(self.num_nodes):
            if nid not in self.tensors:
                continue
            tensor = self.tensors[nid]
            pos_ts = {v: k for k, v in self.pos_by_ts[nid].items()}
            for t in self._valid_positions(nid):
                idx.append((nid, t, pos_ts[t]))
        if len(idx) > MAX_WINDOW_SAMPLES:
            chosen = np.random.choice(len(idx), MAX_WINDOW_SAMPLES, replace=False)
            idx = [idx[i] for i in chosen]
        return idx

    def build_forecast_indices(self, block_start, block_end):
        """Samples whose base timestamp is in [block_start, block_end) AND whose
        FUT targets are all present in the buffer (so we can score them)."""
        fc = []
        for nid in range(self.num_nodes):
            if nid not in self.tensors:
                continue
            pos_ts = {v: k for k, v in self.pos_by_ts[nid].items()}
            for t in self._valid_positions(nid):
                base_ts = pos_ts[t]
                if block_start <= base_ts < block_end:
                    fc.append((nid, t, base_ts))
        return fc


# =====================================================
# DATASET (reads from a RollingBuffer snapshot)
# =====================================================
class ReplayDataset(Dataset):
    def __init__(self, indices, buf):
        self.indices = indices
        self.buf = buf

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        nid, t, ts = self.indices[idx]
        arr = self.buf.tensors[nid]
        phase = self.buf.phase[nid]

        hist_va = arr[t - HIST:t, 0:2].reshape(-1)
        base_pq = arr[t, 2:4] if USE_CURRENT_PQ else arr[t - 1, 2:4]

        # lag rows by timestamp lookup (None if missing → zero-flag in assembler)
        day_pos = self.buf.pos_by_ts[nid].get(ts - DAY_LAG_SEC, None)
        week_pos = self.buf.pos_by_ts[nid].get(ts - WEEK_LAG_SEC, None)
        day_row = arr[day_pos] if day_pos is not None else None
        week_row = arr[week_pos] if week_pos is not None else None

        time_feat = arr[t, 4:6]

        X = assemble_input_vector(hist_va, base_pq, day_row, week_row, phase, time_feat)
        y = arr[t + 1:t + 1 + FUT, 0:2].reshape(-1)
        meta = {"nid": int(nid), "t": int(t), "Timestamp": int(ts)}
        return X, y, torch.tensor(nid, dtype=torch.long), meta


# =====================================================
# MODEL
# =====================================================
class DNN(nn.Module):
    def __init__(self, num_nodes, dropout_p=0.03):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, 8)
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM + 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, OUTPUT_DIM),
        )
    def forward(self, x, nid):
        emb = self.node_emb(nid)
        return self.net(torch.cat([x, emb], dim=1))


def build_model(num_nodes):
    model = DNN(num_nodes=num_nodes, dropout_p=DROPOUT_P).to(device)
    # --- FUTURE HOOK (Change 3): model.share_memory() before spawning the
    #     forecast process so weight updates propagate without files. ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.6)
    criterion = nn.MSELoss()
    scaler_amp = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
    return model, optimizer, scheduler, criterion, scaler_amp


def make_loader(indices, buf, shuffle):
    ds = ReplayDataset(indices, buf)
    return DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )


# =====================================================
# TRAIN / FORECAST (unchanged logic; buffer-backed)
# =====================================================
def train_block(model, optimizer, scheduler, criterion, scaler_amp,
                train_idx, val_idx, buf, block_id, log=print):
    train_loader = make_loader(train_idx, buf, shuffle=True)
    val_loader = make_loader(val_idx, buf, shuffle=False)
    best_val = float("inf"); patience = 2; wait = 0
    for epoch in range(EPOCHS_PER_BLOCK):
        model.train(); total_loss = 0.0
        for X, y, nid, _ in train_loader:
            X = X.to(device); y = y.to(device); nid = nid.long().to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                loss = criterion(model(X, nid), y)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer); scaler_amp.update()
            total_loss += loss.item()
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for X, y, nid, _ in val_loader:
                X = X.to(device); y = y.to(device); nid = nid.long().to(device)
                val_loss += criterion(model(X, nid), y).item()
        scheduler.step()
        log(f"  Epoch {epoch+1:02d} | train={total_loss:.4f} | val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss; wait = 0
        else:
            wait += 1
            if wait >= patience:
                log("  Early stopping."); break


def build_forecast_json(preds, nids, base_ts, buf, base_time=None, simulation_id=None):
    """Assemble the per-node forecast JSON for ONE base timestamp.
    This is the output unit that will later be published to the GridAPPS-D bus
    (one structure per forecast actually run). Physical units; epoch seconds.

    Node identity: internally the NN uses a single combined key (e.g. "632.1").
    On output we split it back on the LAST dot into separate ConnectivityNode
    ("632") and phase ("1") fields — matching the GridAPPS-D publish
    convention. Phase is emitted AS-IS (no mapping); dots are only separators.

    If base_time is None, uses the LATEST (max) scorable base timestamp present.
    """
    if base_time is None:
        base_time = int(base_ts.max())

    row_mask = (base_ts == base_time)
    sel_preds = preds[row_mask]
    sel_nids = nids[row_mask]

    forecast_times = [int(base_time + TS_INCREMENT_SEC * (k + 1)) for k in range(FUT)]

    nodes_out = {}
    for row, nid in zip(sel_preds, sel_nids):
        node_key = buf.id_to_node[int(nid)]
        if "." in node_key:
            cn, phase = node_key.rsplit(".", 1)
        else:
            cn, phase = node_key, ""
        V_series = buf.sc_V.inverse(row[0::2])
        ang_series = buf.sc_ang.inverse(row[1::2])
        nodes_out[node_key] = {
            "ConnectivityNode": cn,
            "phase": phase,
            "V": [float(v) for v in V_series],
            "Angle": [float(a) for a in ang_series],
        }

    return {
        "timestamp": int(base_time),     # top-level: generic ADMS field
        "simulation_id": simulation_id,  # top-level: generic ADMS field
        "Forecast": {                    # forecast-specific payload nested here
            "step_sec": TS_INCREMENT_SEC,
            "horizon": FUT,
            "forecast_times": forecast_times,
            "nodes": nodes_out,
        },
    }


def utc_str(epoch_sec):
    return datetime.fromtimestamp(epoch_sec, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# =====================================================
# End from forecaster_single.py
# =====================================================

# =====================================================
# LOGGING (per-process: each process configures its own file + console)
# =====================================================
def setup_logger(name, logfile):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(processName)s] %(message)s",
                            datefmt="%H:%M:%S")
    fh = logging.FileHandler(logfile, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# =====================================================
# QUEUE HELPERS
# =====================================================
def drain_latest(q):
    """Consumer-side drain: return (latest_non_DONE_item, saw_done).
    Keeps only the newest real item; never discards a DONE marker.
    Non-blocking; returns (None, False) if the queue was empty."""
    latest = None
    saw_done = False
    try:
        while True:
            item = q.get_nowait()
            if isinstance(item, str) and item == DONE:
                saw_done = True
            else:
                latest = item
    except queue.Empty:
        pass
    return latest, saw_done


def drain_all(q):
    """Drain ALL pending items in order. Returns (records_list, saw_done).
    Used for the forecaster's keep-all data queue so history stays gapless."""
    records = []
    saw_done = False
    try:
        while True:
            item = q.get_nowait()
            if isinstance(item, str) and item == DONE:
                saw_done = True
            else:
                records.append(item)
    except queue.Empty:
        pass
    return records, saw_done


# =====================================================
# FEEDER PROCESS
# Source is either the GridAPPS-D bus (gappsd_simid set) or a file (simid None).
# Both drivers funnel records through the shared per-record imputer step() and
# a shared emit() that enqueues the resulting burst to BOTH data queues.
# =====================================================
def feeder_proc(train_data_q, fc_data_q, sim_done, gappsd_simid, data_path):
    log = setup_logger("feeder", FEEDER_LOG)

    # Per-record imputer step + cadence-guard (shared by whichever driver runs).
    step = make_imputer(TS_INCREMENT_SEC)

    # counters (shared)
    n_real = 0
    n_imp = 0
    last_ts = None

    def emit(record, pace=False):
        """Run one record through the imputer and enqueue the resulting burst
        (imputed records + the real one) onto both queues. Imputed records go
        out back-to-back; optional pacing applies once, after the real record."""
        nonlocal n_real, n_imp, last_ts
        for out_rec in step(record):
            train_data_q.put(out_rec)
            fc_data_q.put(out_rec)
            if out_rec.get("_imputed", False):
                n_imp += 1
            else:
                n_real += 1
                last_ts = int(out_rec["timestamp"])
                #if n_real % 500 == 0:
                if n_real % 60 == 0:
                    log.info(f"fed {n_real} real (+{n_imp} imputed) "
                             f"| latest_ts={utc_str(last_ts)}")
        if pace and FEED_RATE_HZ > 0:
            time.sleep(1.0 / FEED_RATE_HZ)   # pace only on real estimates (file driver)

    # ---------- BUS DRIVER (GridAPPS-D) ----------
    if gappsd_simid is not None:
        from gridappsd import GridAPPSD
        from gridappsd.topics import service_output_topic

        os.environ['GRIDAPPSD_APPLICATION_ID'] = 'state-forecaster'
        os.environ['GRIDAPPSD_APPLICATION_STATUS'] = 'STARTED'
        os.environ['GRIDAPPSD_USER'] = 'app_user'
        os.environ['GRIDAPPSD_PASSWORD'] = '1234App'

        keepLoopingFlag = True

        def estimateCallback(header, message):
            nonlocal keepLoopingFlag
            if 'processStatus' in message:
                if message['processStatus'] == "COMPLETE":
                    log.info("Got processStatus COMPLETE message")
                    keepLoopingFlag = False
                return
            # unwrap: message -> message -> Estimate -> SvEstVoltages
            est = message['message']['Estimate']
            ts = int(est['timeStamp'])
            nodes = sv_entries_to_nodes(est['SvEstVoltages'])
            # build internal record: {"timestamp", "nodes": {key: {P,Q,V,Angle}}}
            emit({"timestamp": ts, "nodes": nodes})   # no pacing on bus

        gapps = GridAPPSD(gappsd_simid)
        assert gapps.connected
        gapps.subscribe(service_output_topic('state-estimator', gappsd_simid),
                        estimateCallback)
        log.info(f"FEEDER start | GridAPPS-D simid={gappsd_simid} "
                 f"| increment={TS_INCREMENT_SEC}s")

        while keepLoopingFlag:
            time.sleep(FEEDER_POLL_SEC)

        sim_done.set()
        train_data_q.put(DONE)
        fc_data_q.put(DONE)
        train_data_q.cancel_join_thread()
        fc_data_q.cancel_join_thread()
        log.info(f"FEEDER done | total={n_real} real + {n_imp} imputed "
                 f"| last_ts={utc_str(last_ts) if last_ts else 'n/a'} | sent DONE")
        return

    # ---------- FILE DRIVER ----------
    log.info(f"FEEDER start | path={data_path} | rate={FEED_RATE_HZ} Hz "
             f"| increment={TS_INCREMENT_SEC}s")
    for record in read_json_records(data_path):
        emit(record, pace=True)

    sim_done.set()
    train_data_q.put(DONE)
    fc_data_q.put(DONE)
    log.info(f"FEEDER done | total={n_real} real + {n_imp} imputed "
             f"| last_ts={utc_str(last_ts) if last_ts else 'n/a'} | sent DONE")
    train_data_q.cancel_join_thread()
    fc_data_q.cancel_join_thread()


# =====================================================
# TRAINER PROCESS  (STUB body)
# Keep-all FIFO consume → own RollingBuffer. Lazy init on first record.
# At each block boundary: push a model snapshot (CPU state_dict + version).
# Real training loop arrives in Piece 2.
# =====================================================
def snapshot_to_bytes(model, buf, version):
    """Serialize weights + scaler state to a bytes blob for cross-process
    transport. The scalers travel WITH the model so the forecaster normalizes
    inputs exactly as the trainer did."""
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    buf_io = io.BytesIO()
    torch.save({"weights": sd,
                "scaler_state": extract_scaler_state(buf)}, buf_io)
    return {"version": version, "blob": buf_io.getvalue()}


def trainer_train_block(buf, model, optimizer, scheduler, criterion, scaler_amp,
                        block_start, block_end, block_id, version, model_q, log):
    """Real per-block training (mirrors Step B process_block, sans forecast)."""
    # 1) causal scaler update using ONLY this block's raw values
    buf.update_scalers_with_block(block_start, block_end)
    # 2) rebuild normalized tensors from raw buffers with updated scalers
    buf.rebuild_normalized()
    # 3) build training set from the whole retained buffer (subsampled)
    train_idx = buf.build_training_indices()
    np.random.shuffle(train_idx)
    n_val = int(len(train_idx) * VAL_FRACTION)
    val_idx = train_idx[:n_val]
    tr_idx = train_idx[n_val:]
    log.info(f"[TRAIN] block {block_id} {utc_str(block_start)} → {utc_str(block_end)} "
             f"| train={len(tr_idx)} val={len(val_idx)}")
    if len(tr_idx) == 0:
        log.info(f"  block {block_id}: no training samples — pushing current weights")
    else:
        # train_block logs epoch/loss via log.info → lands in trainer.log
        train_block(model, optimizer, scheduler, criterion, scaler_amp,
                    tr_idx, val_idx, buf, block_id, log=log.info)
    # 4) push the TRAINED snapshot (bytes transport)
    model_q.put(snapshot_to_bytes(model, buf, version))
    log.info(f"pushed model snapshot v{version}")
    # 5) evict rows older than the retention horizon
    buf.evict_old()


def trainer_proc(train_data_q, model_q, sim_done):
    log = setup_logger("trainer", TRAINER_LOG)
    log.info("TRAINER start")
    buf = None
    model = optimizer = scheduler = criterion = scaler_amp = None
    block_start = None
    block_end = None
    block_id = 0
    version = 0

    while True:
        if sim_done.is_set():
            model_q.put(DONE)
            log.info("TRAINER received DONE event → sent DONE to model queue → exit")
            return

        item = train_data_q.get()  # blocking; keep-all FIFO
        if isinstance(item, str) and item == DONE:
            model_q.put(DONE)
            train_data_q.cancel_join_thread()
            model_q.cancel_join_thread()
            log.info("TRAINER received DONE on queue → sent DONE to model queue → exit")
            return

        record = item
        ts = int(record["timestamp"])

        # lazy init from first record (keep ALL build_model returns now)
        if buf is None:
            node_names = sorted(record["nodes"].keys())
            buf = RollingBuffer(node_names)
            model, optimizer, scheduler, criterion, scaler_amp = build_model(buf.num_nodes)
            block_start = ts
            block_end = block_start + BLOCK_SEC
            log.info(f"lazy-init | {buf.num_nodes} nodes | first_ts={utc_str(ts)} "
                     f"| block_end={utc_str(block_end)}")

        # close out any completed block(s) before ingesting this record
        while ts >= block_end:
            if sim_done.is_set():
                model_q.put(DONE)
                train_data_q.cancel_join_thread()
                model_q.cancel_join_thread()
                log.info("TRAINER received DONE event mid-catchup "
                         "→ sent DONE to model queue → exit")
                return
            block_id += 1
            version += 1
            trainer_train_block(buf, model, optimizer, scheduler, criterion,
                                scaler_amp, block_start, block_end, block_id,
                                version, model_q, log)
            block_start = block_end
            block_end = block_start + BLOCK_SEC

        buf.append_record(record)


# =====================================================
# FORECASTER PROCESS  (STUB body)
# Latest-only DATA consume → own RollingBuffer. Lazy init on first record.
# Each loop: check MODEL queue FIRST (load newest, honor DONE), then take
# latest data record and "forecast" (stub logs). Real forecast+JSON in Piece 3.
# =====================================================
FORECAST_LOG_EVERY = 200   # stub: log 1 of every N forecasts (avoids per-poll flood)

def normalize_row(scalers, V, ang, P, Q, ts):
    """Normalize one raw row into [V,ang,P,Q,sin,cos] float32 using given scalers.
    scalers = (sc_V, sc_ang, sc_P, sc_Q)."""
    sc_V, sc_ang, sc_P, sc_Q = scalers
    Vn = float(sc_V.transform(np.array([V], dtype=np.float64))[0])
    an = float(sc_ang.transform(np.array([ang], dtype=np.float64))[0])
    Pn = float(sc_P.transform(np.array([P], dtype=np.float64))[0])
    Qn = float(sc_Q.transform(np.array([Q], dtype=np.float64))[0])
    s, c = time_features(int(ts))
    return np.array([Vn, an, Pn, Qn, s, c], dtype=np.float32)


class NormStore:
    """Forecaster-side normalized history. Per node: an ordered deque of
    (ts, norm_row) for the position-based HIST window, and a {ts: norm_row}
    dict for O(1) lag lookups. Incremental on append; full rebuild only when
    scalers change (snapshot arrival)."""
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.rows = {nid: deque() for nid in range(num_nodes)}     # (ts, norm_row)
        self.by_ts = {nid: {} for nid in range(num_nodes)}         # ts -> norm_row

    def append(self, nid, ts, norm_row):
        self.rows[nid].append((ts, norm_row))
        self.by_ts[nid][ts] = norm_row

    def evict_before(self, cutoff_ts):
        for nid in range(self.num_nodes):
            dq = self.rows[nid]
            bt = self.by_ts[nid]
            while dq and dq[0][0] < cutoff_ts:
                old_ts, _ = dq.popleft()
                bt.pop(old_ts, None)

    def rebuild_from_raw(self, buf, scalers):
        """Full re-normalize of everything currently in buf.raw with new scalers.
        Called once per snapshot arrival (rare)."""
        for nid in range(self.num_nodes):
            self.rows[nid].clear()
            self.by_ts[nid].clear()
            for (ts, V, ang, P, Q) in buf.raw[nid]:
                nr = normalize_row(scalers, V, ang, P, Q, ts)
                self.rows[nid].append((ts, nr))
                self.by_ts[nid][ts] = nr


def forecast_latest(model, store, buf, latest_ts, log):
    """Produce a forecast for base timestamp latest_ts across all nodes.
    Returns a preds array + parallel nid list for build_forecast_json, or None
    if no node has enough history yet."""
    X_list, nid_list = [], []

    for nid in range(store.num_nodes):
        dq = store.rows[nid]
        if len(dq) < HIST + 1:
            continue  # not enough consecutive history yet

        # last HIST+1 rows: the final one is the base (latest_ts), the prior HIST are history
        recent = list(islice(reversed(dq), 0, HIST + 1))  # newest-first, length HIST+1
        recent.reverse()                                   # oldest-first
        base_ts, base_row = recent[-1]
        if base_ts != latest_ts:
            continue  # this node has no row exactly at latest_ts (gap) — skip it

        hist_rows = recent[:HIST]                          # HIST rows before base
        hist_va = torch.tensor(
            np.concatenate([r[0:2] for (_, r) in hist_rows]), dtype=torch.float32)

        base_np = base_row if USE_CURRENT_PQ else recent[-2][1]
        base_pq = torch.tensor(base_np[2:4], dtype=torch.float32)

        day_np = store.by_ts[nid].get(latest_ts - DAY_LAG_SEC, None)
        week_np = store.by_ts[nid].get(latest_ts - WEEK_LAG_SEC, None)
        day_row = torch.tensor(day_np, dtype=torch.float32) if day_np is not None else None
        week_row = torch.tensor(week_np, dtype=torch.float32) if week_np is not None else None

        time_feat = torch.tensor(base_row[4:6], dtype=torch.float32)
        phase = buf.phase[nid]

        X = assemble_input_vector(hist_va, base_pq, day_row, week_row, phase, time_feat)
        X_list.append(X)
        nid_list.append(nid)

    if not X_list:
        return None

    X_batch = torch.stack(X_list).to(device)
    nid_batch = torch.tensor(nid_list, dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        preds = model(X_batch, nid_batch).cpu().numpy()

    nids = np.array(nid_list)
    base_ts_arr = np.full(len(nid_list), latest_ts, dtype=np.int64)
    return preds, nids, base_ts_arr


def forecaster_proc(fc_data_q, model_q, gappsd_simid):
    log = setup_logger("forecaster", FORECASTER_LOG)
    log.info("FORECASTER start")

    # GridAPPS-D connection for PUBLISHING forecasts (no subscription here —
    # the forecaster only produces output). Created INSIDE this child process
    # (spawn) since connection objects don't pickle across the process boundary.
    # `gapps` stays None for file-based runs (gappsd_simid is None).
    gapps = None
    if gappsd_simid is not None:
        from gridappsd import GridAPPSD
        from gridappsd.topics import service_output_topic
        os.environ['GRIDAPPSD_APPLICATION_ID'] = 'state-forecaster'
        os.environ['GRIDAPPSD_APPLICATION_STATUS'] = 'STARTED'
        os.environ['GRIDAPPSD_USER'] = 'app_user'
        os.environ['GRIDAPPSD_PASSWORD'] = '1234App'
        gapps = GridAPPSD(gappsd_simid)
        assert gapps.connected
        log.info(f"FORECASTER connected to GridAPPS-D simid={gappsd_simid} "
                 f"for publishing forecasts")
        publish_to_topic = service_output_topic('state-forecaster', gappsd_simid)

    buf = None
    model = None
    store = None          # NormStore (incremental normalized rows)
    scalers = None        # (sc_V, sc_ang, sc_P, sc_Q) — refs into buf's scalers
    have_scalers = False  # True once first snapshot applied
    current_version = 0
    pending_snap = None
    data_done = False
    fc_count = 0
    warmup_logged = False

    # --- deferred live MAE scoring state ---
    score_armed = False        # arm on snapshot adoption
    pending = None             # {"remaining": set(ts), "pred": {ts: {node: (V, Ang)}},
                               #  "abs_v": [...], "abs_a": [...], "n": 0, "version": int}

    # Forecast-output file: truncate any existing file at startup, then append
    # one JSON line per published forecast (open/append/close per write --
    # durable and handle held across the run). None disables it.
    # Opened once here, written per forecast, closed at exit
    if FORECAST_OUTPUT_JSONL:
        with open(FORECAST_OUTPUT_JSONL, "w"):
            pass # create/truncate to empty
        log.info(f"Writing published forecasts to {FORECAST_OUTPUT_JSONL}")

    def score_pending(record):
        """Deferred MAE: if this real estimate's timestamp matches a pending
        forecast step, accumulate abs error (nodes present in both)."""
        nonlocal pending
        if pending is None:
            return
        ts = int(record["timestamp"])
        if ts not in pending["remaining"]:
            return
        pred_at = pending["pred"][ts]
        for node, vals in record["nodes"].items():
            p = pred_at.get(node)
            if p is None:
                continue                      # node not in forecast — skip (defensive)
            pending["abs_v"] += abs(p[0] - vals["V"])
            pending["abs_a"] += abs(p[1] - vals["Angle"])
            pending["n"] += 1
        pending["remaining"].discard(ts)
        if not pending["remaining"]:          # all horizon steps collected
            n = max(pending["n"], 1)
            log.info(f"[MAE] v{pending['version']} base_time="
                     f"{utc_str(pending['base_time'])} | "
                     f"Voltage MAE (pu): {pending['abs_v']/n:.6f} | "
                     f"Angle MAE (rad): {pending['abs_a']/n:.6f} "
                     f"({pending['n']} node-steps)")
            pending = None

    def ingest(record):
        """Append raw to buf; if scalers known, incrementally normalize into store."""
        ts = int(record["timestamp"])
        buf.append_record(record)   # updates buf.raw + buf.newest_ts
        if have_scalers:
            for node_name, vals in record["nodes"].items():
                nid = buf.node_to_id.get(node_name)
                if nid is None:
                    continue
                P = vals["P"] if vals["P"] is not None else 0.0
                Q = vals["Q"] if vals["Q"] is not None else 0.0
                nr = normalize_row(scalers, vals["V"], vals["Angle"], P, Q, ts)
                store.append(nid, ts, nr)

    def evict():
        if buf.newest_ts is None:
            return
        cutoff = buf.newest_ts - RETENTION_SEC
        buf.evict_old()                 # raw
        if store is not None:
            store.evict_before(cutoff)  # normalized

    while True:
        # 1) MODEL QUEUE FIRST: adopt newest snapshot (weights + scalers), honor DONE.
        snap, model_saw_done = drain_latest(model_q)
        if snap is not None:
            if buf is None:
                pending_snap = snap   # arrived before first data; apply after init
            else:
                blob = torch.load(io.BytesIO(snap["blob"]), map_location="cpu")
                model.load_state_dict(blob["weights"])
                apply_scaler_state(buf, blob["scaler_state"])
                store.rebuild_from_raw(buf, scalers)   # full re-normalize (rare)
                have_scalers = True
                current_version = snap["version"]
                if COMPUTE_LIVE_MAE:
                    score_armed = True   # score the NEXT forecast made under this version
                log.info(f"adopted snapshot v{current_version} "
                         f"→ scalers updated, store re-normalized "
                         f"({sum(len(store.rows[n]) for n in range(store.num_nodes))} rows)")

        # 2) DATA QUEUE: drain ALL (keep-all, gapless), ingest in order.
        records, data_saw_done = drain_all(fc_data_q)
        if data_saw_done:
            data_done = True

        latest_ts = None
        latest_imputed = False
        for record in records:
            if buf is None:
                # lazy init from first record (always REAL: no leading
                # imputation is ever emitted, so the first record is real)
                node_names = sorted(record["nodes"].keys())
                buf = RollingBuffer(node_names)
                model, *_ = build_model(buf.num_nodes)
                store = NormStore(buf.num_nodes)
                scalers = (buf.sc_V, buf.sc_ang, buf.sc_P, buf.sc_Q)
                log.info(f"lazy-init | {buf.num_nodes} nodes | first_ts={utc_str(int(record['timestamp']))}")
                if pending_snap is not None:
                    blob = torch.load(io.BytesIO(pending_snap["blob"]),
                                      map_location="cpu")
                    model.load_state_dict(blob["weights"])
                    apply_scaler_state(buf, blob["scaler_state"])
                    have_scalers = True
                    current_version = pending_snap["version"]
                    log.info(f"applied held snapshot v{current_version}")
                    pending_snap = None
            # Ingest EVERY record (imputed + real) to keep history gapless.
            ingest(record)
            latest_ts = int(record["timestamp"])
            latest_imputed = bool(record.get("_imputed", False))
            if COMPUTE_LIVE_MAE and not latest_imputed:
                score_pending(record)      # deferred MAE on real estimates only

        if records:
            evict()

        # 3) FORECAST the latest timestamp — but ONLY if it is a REAL estimate.
        # Imputed (interpolated) data is ingested for gapless history but must
        # not trigger a published forecast. Because the feeder emits each gap as
        # a [imp, ..., real] burst, the latest drained record is normally real;
        # if a drain lands mid-burst on an imputed record, we simply skip this
        # cycle and forecast on the real record that arrives next cycle.
        if latest_ts is not None and not latest_imputed:      # NEW: skip if imputed
            if have_scalers:
                result = forecast_latest(model, store, buf, latest_ts, log)
                if result is not None:
                    preds, nids, base_ts_arr = result
                    fc_count += 1
                    fc_json = build_forecast_json(preds, nids, base_ts_arr, buf,
                                                  base_time=latest_ts,
                                                  simulation_id=gappsd_simid)

                    if gapps is not None:
                        gapps.send(publish_to_topic, json.dumps(fc_json))
                    if FORECAST_OUTPUT_JSONL:
                        with open(FORECAST_OUTPUT_JSONL, "a") as f:
                            f.write(json.dumps(fc_json) + "\n")

                    if fc_count == 1 or fc_count % FORECAST_LOG_EVERY == 0:
                        log.info(f"[FORECAST] #{fc_count} base_time={utc_str(latest_ts)} "
                                 f"using model v{current_version} | {len(fc_json['Forecast']['nodes'])} nodes")
                        #log.info(json.dumps(fc_json))

                    if COMPUTE_LIVE_MAE and score_armed:
                        pred = {}
                        for k in range(FUT):
                            ft = fc_json["Forecast"]["forecast_times"][k]
                            pred[ft] = {
                                node: (nd["V"][k], nd["Angle"][k])
                                for node, nd in fc_json["Forecast"]["nodes"].items()
                            }
                        pending = {
                            "remaining": set(pred.keys()),
                            "pred": pred,
                            "abs_v": 0.0, "abs_a": 0.0, "n": 0,
                            "version": current_version,
                            "base_time": latest_ts,
                        }
                        score_armed = False
                        log.info(f"[MAE] armed: scoring forecast v{current_version} "
                                 f"base_time={utc_str(latest_ts)} over {FUT} steps")

            else:
                if not warmup_logged:
                    log.info(f"latest_ts={utc_str(latest_ts)} | no model yet "
                             f"(warm-up) — skipping forecasts until first snapshot")
                    warmup_logged = True

        # 4) shutdown when BOTH streams exhausted.
        if data_done and model_saw_done:
            log.info(f"FORECASTER received DONE on both queues → exit "
                     f"(total forecasts: {fc_count}, final model v{current_version})")

            if gapps is not None:
                done_json = {"simulation_id": gappsd_simid,
                             "processStatus": "COMPLETE"}
                gapps.send(publish_to_topic, json.dumps(done_json))

            fc_data_q.cancel_join_thread()
            model_q.cancel_join_thread()
            return

        # 5) avoid busy-spin when idle
        if not records and snap is None:
            time.sleep(FORECASTER_POLL_SEC)


# =====================================================
# MAIN — spawn the three processes, wire the queues.
# =====================================================
def main():
    # GDB 7/20/26: GriAPPS-D simulation ID is first command line argument
    gappsd_simid = None
    if len(sys.argv) > 1:
      gappsd_simid = sys.argv[1]

    mp.set_start_method("spawn", force=True)  # required for CUDA + multiprocessing

    # --- Queues ---
    # Trainer data: keep-ALL FIFO (unbounded). Every record must be retained
    #   so the trainer's RollingBuffer has no gaps.
    # Forecaster data: keep-ALL FIFO (unbounded). Every record must be retained
    # Model: trainer PUTs snapshots; forecaster GETs. Latest-only via
    #   drain_latest on the consumer side (unbounded; snapshots are infrequent).
    train_data_q = mp.Queue()
    fc_data_q = mp.Queue()
    model_q = mp.Queue()
    sim_done = mp.Event()

    procs = [
        mp.Process(target=feeder_proc,
                   args=(train_data_q, fc_data_q, sim_done,
                         gappsd_simid, JSON_PATH),
                   name="feeder"),
        mp.Process(target=trainer_proc,
                   args=(train_data_q, model_q, sim_done),
                   name="trainer"),
        mp.Process(target=forecaster_proc,
                   args=(fc_data_q, model_q, gappsd_simid),
                   name="forecaster"),
    ]

    print(f"[MAIN] spawning {len(procs)} processes "
          f"(feeder rate={FEED_RATE_HZ} Hz). Logs: "
          f"{FEEDER_LOG}, {TRAINER_LOG}, {FORECASTER_LOG}")
    print("[MAIN] tip: `tail -f trainer.log` and `tail -f forecaster.log` "
          "in separate terminals.")

    for p in procs:
        p.start()

    try:
        # Normal shutdown: feeder finishes → sends DONE → trainer finalizes and
        # sends DONE to model queue → forecaster sees DONE on both → all exit.
        for p in procs:
            p.join()
        bad = [p for p in procs if p.exitcode not in (0, None)]
        if bad:
            print(f"[MAIN] processes exited with errors: "
                  f"{[(p.name, p.exitcode) for p in bad]}")
        else:
            print("[MAIN] all processes exited cleanly.")
    except KeyboardInterrupt:
        # Ctrl-C: tear down children so we don't leave orphans.
        print("\n[MAIN] KeyboardInterrupt → terminating child processes...")
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
        print("[MAIN] shutdown complete.")
    finally:
        # Report any non-zero exit codes (a crashed child shows up here).
        for p in procs:
            if p.exitcode not in (0, None):
                print(f"[MAIN] WARNING: {p.name} exited with code {p.exitcode}")


if __name__ == "__main__":
    main()
