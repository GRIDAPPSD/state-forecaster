#!/usr/bin/python3
# =====================================================
# Step B: streaming ingestion (one timestamp at a time),
# rolling buffer, incremental scalers, forecast-then-train.
# pandas removed from the data path (stdlib datetime only).
# =====================================================
import json
import math
from collections import deque, OrderedDict
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =====================================================
# CONFIG  (explicit, not implicit — designed for configurability)
# =====================================================
#JSON_PATH = "results_data_forecasting_13_real.json"
JSON_PATH = "results_data_forecasting_13_5min_real.json"
#JSON_PATH = "results_data_forecasting_123_real.json"
#JSON_PATH = "gappy_13.json"

# --- streaming cadence ---
#TS_INCREMENT_SEC = 60          # timestamp spacing in the input stream (1 min)
TS_INCREMENT_SEC = 300          # timestamp spacing in the input stream (5 min)
#TS_INCREMENT_SEC = 900          # timestamp spacing in the input stream (15 min)

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

def _pq_value(x):
    """Coerce a P or Q field to float. The real State Estimator publishes the
    string "NA" for buses without a P/Q estimate (e.g. SOURCEBUS); map those to
    0.0 (matches the prior simplified-file handling). None is also treated as 0.0."""
    if x == "NA" or x is None:
        return 0.0
    return float(x)

def read_json_records(path):
    """Yield one internal record per line, parsed from the GridAPPS-D State
    Estimator publish format (SvEstVoltages).

    Input line (real format):
        {"SvEstVoltages": [{"ConnectivityNode": "632", "phase": "1",
                             "P": .., "Q": .., "v": .., "angle": ..}, ...],
         "timeStamp": 1700000000}

    Yields internal record (unchanged downstream shape):
        {"timestamp": 1700000000,
         "nodes": {"632.1": {"P": .., "Q": .., "V": .., "Angle": ..}, ...}}

    Notes:
      * Internal node key = ConnectivityNode + "." + phase (e.g. "632" + "1"
        -> "632.1"). This single combined key is the NN's node identity
        (Approach 3: combined internally, split back to separate fields only
        at output in build_forecast_json). Phase is used AS-IS (no mapping);
        dots are only ever separators, never part of a ConnectivityNode value.
      * v -> V, angleRad -> Angle
      * P/Q == "NA" -> 0.0 (SOURCEBUS etc.). V and angle are NOT NA-coerced:
        a missing voltage/angle should fail loudly rather than be silently
        zeroed into the history window.
      * variance fields (angleVariance, vVariance) are ignored.
    """
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ts = int(rec["timeStamp"])
            nodes = {}
            for entry in rec["SvEstVoltages"]:
                cn = entry["ConnectivityNode"]
                phase = entry["phase"]
                node_key = f"{cn}.{phase}" if phase != "" else cn
                nodes[node_key] = {
                    "P": _pq_value(entry["P"]),
                    "Q": _pq_value(entry["Q"]),
                    "V": float(entry["v"]),
                    "Angle": float(entry["angleRad"]),
                }
            yield {"timestamp": ts, "nodes": nodes}

def impute_missing_records(source, increment_sec):
    """Wrap a record source to yield a gapless, grid-aligned stream.

    Incoming record timestamps are assumed to be multiples of increment_sec
    (grid-aligned). When two consecutive REAL records are more than one
    increment apart, linearly-interpolated placeholder records are synthesized
    for each missing grid-aligned timestamp and yielded IMMEDIATELY BEFORE the
    real record that triggered them — i.e. a "burst": [imp, imp, ..., real].

    Design points (see project discussion):
      * NO leading imputation: nothing is emitted before the first real record.
        The first real record is effectively "time zero" for forecasting.
      * Imputation only happens once the FORWARD (later) real record is in hand,
        so each imputed record is emitted in the same burst as its trigger.
      * Each imputed record carries "_imputed": True; real records carry
        "_imputed": False. Downstream code that only reads "timestamp"/"nodes"
        is unaffected; the flag lets the forecaster skip forecasting on imputed
        data while still ingesting it to keep history gapless.
      * Interpolation is linear per node for P, Q, V, Angle. (Safe for angle
        here because per-phase angles cluster near 0 / ±2.09 rad, far from the
        ±pi wraparound.)
    """
    prev_ts = None
    prev_nodes = None

    for record in source:
        ts = int(record["timestamp"])

        if prev_ts is not None:
            delta = ts - prev_ts
            if delta <= 0:
                # Non-increasing timestamp: unexpected. Pass through untouched;
                # don't attempt imputation across a non-forward step.
                print(f"[IMPUTE] WARNING: non-increasing timestamp "
                      f"{prev_ts} -> {ts}; passing through without imputation.")
            elif delta % increment_sec != 0:
                # Not grid-aligned: violates the stated assumption. Don't
                # fabricate misaligned rows; pass the real record through and
                # log so it's visible.
                print(f"[IMPUTE] WARNING: gap {delta}s not a multiple of "
                      f"increment {increment_sec}s ({prev_ts} -> {ts}); "
                      f"no imputation for this gap.")
            else:
                gap_steps = delta // increment_sec
                if gap_steps > 1:
                    # Synthesize gap_steps - 1 imputed records along the line
                    # from prev_nodes -> record["nodes"].
                    for k in range(1, gap_steps):
                        frac = k / gap_steps
                        imp_ts = prev_ts + k * increment_sec
                        imp_nodes = {}
                        for node_name, cur_vals in record["nodes"].items():
                            prev_vals = prev_nodes.get(node_name)
                            if prev_vals is None:
                                # Node absent in previous record (shouldn't
                                # happen with fixed node set); fall back to
                                # current values rather than crash.
                                imp_nodes[node_name] = dict(cur_vals)
                            else:
                                imp_nodes[node_name] = {
                                    "P": prev_vals["P"] + frac * (cur_vals["P"] - prev_vals["P"]),
                                    "Q": prev_vals["Q"] + frac * (cur_vals["Q"] - prev_vals["Q"]),
                                    "V": prev_vals["V"] + frac * (cur_vals["V"] - prev_vals["V"]),
                                    "Angle": prev_vals["Angle"] + frac * (cur_vals["Angle"] - prev_vals["Angle"]),
                                }
                        yield {"timestamp": imp_ts,
                               "nodes": imp_nodes,
                               "_imputed": True}

        # Emit the real record (tagged as non-imputed).
        record["_imputed"] = False
        yield record

        prev_ts = ts
        prev_nodes = record["nodes"]

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

def forecast(model, forecast_indices, buf):
    if len(forecast_indices) == 0:
        print("  [FORECAST] No scorable samples in this window — skipping.")
        return None
    loader = make_loader(forecast_indices, buf, shuffle=False)
    model.eval()
    preds, trues, nids, base_ts = [], [], [], []
    with torch.no_grad():
        for X, y, nid, meta in loader:
            X = X.to(device); nid = nid.long().to(device)
            preds.append(model(X, nid).cpu().numpy())
            trues.append(y.numpy())
            nids.append(meta["nid"].numpy())
            base_ts.append(meta["Timestamp"].numpy())
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    nids = np.concatenate(nids)
    base_ts = np.concatenate(base_ts)
    return preds, trues, nids, base_ts

def build_forecast_json(preds, nids, base_ts, buf, base_time=None):
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
    sel_preds = preds[row_mask]          # [num_nodes_at_this_base, FUT*2]
    sel_nids = nids[row_mask]

    forecast_times = [int(base_time + TS_INCREMENT_SEC * (k + 1)) for k in range(FUT)]

    nodes_out = {}
    for row, nid in zip(sel_preds, sel_nids):
        node_key = buf.id_to_node[int(nid)]
        # split combined internal key -> ConnectivityNode + phase (last dot)
        if "." in node_key:
            cn, phase = node_key.rsplit(".", 1)
        else:
            cn, phase = node_key, ""
        V_series = buf.sc_V.inverse(row[0::2])        # FUT voltage-magnitude
        ang_series = buf.sc_ang.inverse(row[1::2])    # FUT angle (rad)
        nodes_out[node_key] = {
            "ConnectivityNode": cn,
            "phase": phase,
            "V": [float(v) for v in V_series],
            "Angle": [float(a) for a in ang_series],
        }

    return {
        "base_time": int(base_time),
        "step_sec": TS_INCREMENT_SEC,
        "horizon": FUT,
        "forecast_times": forecast_times,
        "nodes": nodes_out,
    }

def report_forecast(preds, trues, buf, label):
    print(f"\n====== FORECAST RESULTS [{label}] (Normalized) ======")
    print("MSE:", mean_squared_error(trues, preds))
    print("MAE:", mean_absolute_error(trues, preds))
    # inverse-transform via the buffer's CURRENT incremental scalers
    V_pred = buf.sc_V.inverse(preds[:, 0::2])
    ang_pred = buf.sc_ang.inverse(preds[:, 1::2])
    V_true = buf.sc_V.inverse(trues[:, 0::2])
    ang_true = buf.sc_ang.inverse(trues[:, 1::2])
    print(f"====== FORECAST RESULTS [{label}] (Physical) ======")
    print("Voltage MAE (pu):", mean_absolute_error(V_true, V_pred))
    print("Angle MAE (rad):", mean_absolute_error(ang_true, ang_pred))

# =====================================================
# STREAMING RUN LOOP
# Consumes records one timestamp at a time from the record source.
# Block boundaries are aligned to the first timestamp seen. When a block
# completes: update scalers -> rebuild normalized tensors -> (forecast the
# just-completed block if past warm-up) -> train on the buffer -> evict old.
# =====================================================
def utc_str(epoch_sec):
    return datetime.fromtimestamp(epoch_sec, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def process_block(model, opt, sched, crit, amp, buf,
                  block_start, block_end, block_id):
    # 1) causal scaler update using ONLY this block's raw values
    buf.update_scalers_with_block(block_start, block_end)
    # 2) rebuild normalized tensors from raw buffers with updated scalers
    buf.rebuild_normalized()

    # 3) forecast-then-train (single-process scaffolding).
    #    Warm-up: no forecast until the model has trained on >= MIN blocks.
    if block_id > MIN_BLOCKS_BEFORE_FORECAST:
        fc_idx = buf.build_forecast_indices(block_start, block_end)
        print(f"[FORECAST] Block {block_id} "
              f"{utc_str(block_start)} → {utc_str(block_end)} | samples={len(fc_idx)}")
        result = forecast(model, fc_idx, buf)
        if result is not None:
            preds, trues, nids, base_ts = result
            report_forecast(preds, trues, buf,
                            label=f"Block {block_id} ({utc_str(block_start)})")
            # --- per-node forecast JSON for the latest scorable base timestamp ---
            fc_json = build_forecast_json(preds, nids, base_ts, buf)
            print(f"\n------ FORECAST JSON (base_time={fc_json['base_time']} "
                  f"= {utc_str(fc_json['base_time'])}, "
                  f"{len(fc_json['nodes'])} nodes, horizon={fc_json['horizon']}) ------")
            print(json.dumps(fc_json, indent=2))

    else:
        print(f"[WARM-UP] Block {block_id}: train-only "
              f"({utc_str(block_start)} → {utc_str(block_end)}), no forecast yet.")

    # 4) train cumulatively on everything currently retained in the buffer
    train_idx = buf.build_training_indices()
    np.random.shuffle(train_idx)
    n_val = int(len(train_idx) * VAL_FRACTION)
    val_idx = train_idx[:n_val]
    tr_idx = train_idx[n_val:]
    print(f"[TRAIN] Block {block_id} | train={len(tr_idx)} val={len(val_idx)}")
    if len(tr_idx) == 0:
        print(f"  Block {block_id}: no training samples "
              f"(insufficient data: need > HIST+FUT={HIST+FUT} timestamps) — skipping train.")
    else:
        train_block(model, opt, sched, crit, amp, tr_idx, val_idx, buf, block_id)

    # 5) evict rows older than the retention horizon
    buf.evict_old()

def run(data_path):
    source = impute_missing_records(read_json_records(data_path), TS_INCREMENT_SEC)

    # --- discover the fixed node set from the FIRST record ---
    try:
        first = next(source)
    except StopIteration:
        raise ValueError("Empty input stream.")
    node_names = sorted(first["nodes"].keys())
    print(f"Discovered {len(node_names)} phase-nodes from first record.")

    buf = RollingBuffer(node_names)
    model, opt, sched, crit, amp = build_model(buf.num_nodes)

    # block boundaries aligned to the first timestamp
    data_start = int(first["timestamp"])
    block_start = data_start
    block_end = block_start + BLOCK_SEC
    block_id = 0

    print("\n======= STREAMING TRAINING (forecast-then-train per block) =======")
    print("DATA_START :", utc_str(data_start))

    # feed the first record, then the rest
    buf.append_record(first)

    for record in source:
        ts = int(record["timestamp"])
        # close out any completed block(s) before ingesting this record
        while ts >= block_end:
            block_id += 1
            process_block(model, opt, sched, crit, amp, buf,
                          block_start, block_end, block_id)
            block_start = block_end
            block_end = block_start + BLOCK_SEC
        buf.append_record(record)

    # finalize the trailing partial/final block (if it holds any data)
    if buf.newest_ts is not None and buf.newest_ts >= block_start:
        block_id += 1
        process_block(model, opt, sched, crit, amp, buf,
                      block_start, block_end, block_id)

    print("\n======= STREAMING COMPLETE =======")
    return model

if __name__ == "__main__":
    run(JSON_PATH)

