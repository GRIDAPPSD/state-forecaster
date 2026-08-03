"""
forecast_dnn.py — Shared deep-neural-network core for the State Forecaster.

The torch-dependent heart of the app, imported by BOTH the trainer and the
forecaster (they share the model, the rolling buffer, the scalers, and the
input-vector layout). Keeping this in one module guarantees train and predict
build model inputs identically and agree on all model/normalization details.

Contents:
  * Model + feature config (HIST/FUT, lag intervals, retention horizon,
    device/seed setup).
  * time_features / encode_phase / assemble_input_vector (+ INPUT_DIM/OUTPUT_DIM)
    — the input-vector construction (single source of truth for the layout).
  * RunningMinMax / RunningStandardizer — streaming (incremental) scalers, plus
    extract/apply helpers to move their state across the process boundary.
  * RollingBuffer — per-node raw history + normalized tensors + sample indexing.
  * DNN / build_model — the network and its optimizer/scheduler/loss/AMP scaler.
"""

import math
import numpy as np

from datetime import datetime, timezone
from collections import deque

import torch
import torch.nn as nn

DROPOUT_P = 0.03  # training regularization only (no MC dropout)
MAX_WINDOW_SAMPLES = 500_000  # cap on per-block training samples: bounds both
#                               memory and training time (subsample above this)

# --- history / horizon (in SAMPLES, i.e. number of timestamps) ---
# Actual time spans scale with TS_INCREMENT_SEC: e.g. HIST=15 is 15 min at a
# 1-min increment, 75 min at a 5-min increment.
HIST = 15  # past samples used as model input
FUT = 15  # future samples to forecast

# --- lag features (in TIME, seconds) — looked up by timestamp, not position ---
DAY_LAG_SEC = 1 * 24 * 3600  # 1-day lag
WEEK_LAG_SEC = 7 * 24 * 3600  # 1-week lag

RETENTION_DAYS = 10  # rolling buffer horizon (see rationale below)
# Rationale for RETENTION_DAYS:
#   * >= 7 days: distribution load has a WEEKLY trend; a full week must be
#     retained so the 1-week lag feature is populated (not zero-flagged).
#   * The current block's forecast references samples as far back as its START,
#     which needs 7 days *before* the block start -> ~9 days from block end.
#   * Rounded up to 10 so retention is an even multiple of BLOCK_DAYS (5 blocks)
#     and the week-lag never degrades at block boundaries.
#   All heuristics; kept configurable for later evaluation.
RETENTION_SEC = RETENTION_DAYS * 24 * 3600

# Use the injection P,Q at the base (current) row as input; if False, use t-1.
USE_CURRENT_PQ = True

# Fixed seed for reproducibility. NOTE: with the "spawn" start method each child
# process re-imports this module and re-seeds independently.
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

USE_GPU = True
DEVICE = "cuda" if torch.cuda.is_available() and USE_GPU else "cpu"


# =====================================================
# TIME FEATURES (from epoch seconds; no pandas)
# =====================================================
def time_features(epoch_sec):
    """Return (sin_time, cos_time) encoding minute-of-day cyclically.
    Cyclic encoding so 23:59 and 00:00 are adjacent in feature space. UTC to
    match the original (pandas-based) pipeline's timestamp handling."""
    dt = datetime.fromtimestamp(epoch_sec, tz=timezone.utc)
    minute_of_day = dt.hour * 60 + dt.minute
    ang = 2.0 * math.pi * minute_of_day / 1440.0
    return math.sin(ang), math.cos(ang)


def encode_phase(load_node):
    """Return a 3-element one-hot phase vector from a node name's last char.
    Handles both letter (a/b/c) and numeric (1/2/3) phase suffixes, mapping
    each to the same one-hot. Unknown/other -> all zeros. (Single-character
    phase only; multi-char phases like 's1' are out of scope.)"""
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


# Model input/output dimensions. INPUT_DIM's per-line breakdown below MUST stay
# in sync with the concatenation order in assemble_input_vector (kept adjacent
# here on purpose so the two can't drift). Depends on HIST/FUT (defined above).
INPUT_DIM = (
    HIST * 2  # historical V, angle
    + 2  # current or previous P,Q
    + 2  # 1-day lag P,Q
    + 2  # 1-day lag V, angle
    + 1  # 1-day lag availability flag
    + 2  # 1-week lag P,Q
    + 2  # 1-week lag V, angle
    + 1  # 1-week lag availability flag
    + 3  # phase (one-hot)
    + 2  # sin_time, cos_time
)
OUTPUT_DIM = FUT * 2  # forecast V + angle for each of FUT future steps


def assemble_input_vector(
    hist_va, base_pq, day_row, week_row, phase, time_feat
):
    """Single source of truth for the model input layout.
    Used identically by the trainer (ReplayDataset) and the forecaster
    (forecast_latest), guaranteeing both build inputs the same way. The
    concatenation order below mirrors the INPUT_DIM breakdown just above.

    For each lag row: if present, contribute its P,Q and V,angle plus an
    availability flag of 1; if None (that lagged timestamp isn't in the buffer),
    contribute zeros and a flag of 0 — so the model can distinguish "real lag
    data" from "lag unavailable".

    Args (all torch.float32):
        hist_va  : [HIST*2] flattened V,ang of the HIST rows before base
        base_pq  : [2] P,Q of the base row (t if USE_CURRENT_PQ else t-1)
        day_row  : [6] normalized row (V,ang,P,Q,sin,cos) at 1-day lag, or None
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

    return torch.cat(
        [
            hist_va,  # HIST * 2
            base_pq,  # 2
            day_pq,  # 2
            day_va,  # 2
            day_flag,  # 1
            week_pq,  # 2
            week_va,  # 2
            week_flag,  # 1
            phase,  # 3
            time_feat,  # 2
        ]
    )


# =====================================================
# INCREMENTAL SCALERS (streaming replacements for sklearn scalers)
# Updated per-block with that block's raw values, causally (no peeking ahead).
# =====================================================
class RunningMinMax:
    """Streaming MinMax scaler for a single feature; maps values to [0, 1].
    Used for P, Q, and V. Tracks running min/max as blocks arrive."""

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
            # Degenerate range (no data yet, or all-equal): map to zeros.
            return np.zeros_like(x, dtype=np.float64)
        return (x - self.min) / rng

    def inverse(self, x_scaled):
        rng = self.max - self.min
        return x_scaled * rng + self.min


class RunningStandardizer:
    """Streaming standard scaler (zero mean / unit std) for a single feature,
    used for angle. Maintains running mean/variance via Welford/Chan's parallel
    algorithm so per-block updates match a single-pass fit over all data."""

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
        # Chan's parallel merge of the running stats with this batch's stats.
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
    """Snapshot the four scalers' state as plain picklable numbers, for
    inclusion in a model snapshot (trainer side). Paired with apply_scaler_state.
    """
    return {
        "P": (buf.sc_P.min, buf.sc_P.max),
        "Q": (buf.sc_Q.min, buf.sc_Q.max),
        "V": (buf.sc_V.min, buf.sc_V.max),
        "ang": (buf.sc_ang.n, buf.sc_ang.mean, buf.sc_ang.M2),
    }


def apply_scaler_state(buf, state):
    """Restore scaler state (from extract_scaler_state) into a buffer's scalers
    (forecaster side), so the forecaster normalizes exactly as the trainer did.
    """
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
    """Per-node history buffer + normalized-tensor cache + sample indexing.

    Holds each node's recent raw (ts, V, angle, P, Q) rows in time order, bounded
    to RETENTION_SEC. After each block's scalers update, rebuild_normalized()
    produces per-node normalized tensors (and a timestamp->row-position map) that
    the trainer's ReplayDataset and the forecaster read from. Owns the four
    incremental scalers (their state travels with the model snapshot)."""

    def __init__(self, node_names):
        self.node_names = list(node_names)
        self.node_to_id = {n: i for i, n in enumerate(self.node_names)}
        self.id_to_node = {i: n for n, i in self.node_to_id.items()}
        self.num_nodes = len(self.node_names)
        # per node_id: deque of (ts, V, ang, P, Q) in time order
        self.raw = {nid: deque() for nid in range(self.num_nodes)}
        # per-node phase one-hot (static; computed once from the node name)
        self.phase = {
            nid: torch.tensor(encode_phase(n), dtype=torch.float32)
            for nid, n in self.id_to_node.items()
        }
        # rebuilt each block by rebuild_normalized():
        self.tensors = (
            {}
        )  # nid -> float32 [T,6]: V,ang,P,Q,sin,cos (normalized)
        self.pos_by_ts = {}  # nid -> {ts: row index}
        self.newest_ts = None
        # incremental scalers (V/P/Q min-max, angle standardized)
        self.sc_P = RunningMinMax()
        self.sc_Q = RunningMinMax()
        self.sc_V = RunningMinMax()
        self.sc_ang = RunningStandardizer()

    def append_record(self, record):
        """Add one timestamp's worth of node values (raw) to each node's deque."""
        ts = int(record["timestamp"])
        self.newest_ts = ts
        for node_name, vals in record["nodes"].items():
            nid = self.node_to_id.get(node_name)
            if nid is None:
                continue  # node not in the fixed set from the first record
            P = vals["P"] if vals["P"] is not None else 0.0
            Q = vals["Q"] if vals["Q"] is not None else 0.0
            self.raw[nid].append((ts, vals["V"], vals["Angle"], P, Q))

    def evict_old(self):
        """Drop rows older than RETENTION_SEC behind the newest timestamp
        (bounds memory regardless of total run length)."""
        if self.newest_ts is None:
            return
        cutoff = self.newest_ts - RETENTION_SEC
        for nid, dq in self.raw.items():
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    def update_scalers_with_block(self, block_start, block_end):
        """Update the running scalers using ONLY this block's raw values
        (causal: the model never normalizes using data from the future)."""
        Ps, Qs, Vs, As = [], [], [], []
        for dq in self.raw.values():
            for ts, V, A, P, Q in dq:
                if block_start <= ts < block_end:
                    Vs.append(V)
                    As.append(A)
                    Ps.append(P)
                    Qs.append(Q)
        self.sc_V.update(np.asarray(Vs, dtype=np.float64))
        self.sc_ang.update(np.asarray(As, dtype=np.float64))
        self.sc_P.update(np.asarray(Ps, dtype=np.float64))
        self.sc_Q.update(np.asarray(Qs, dtype=np.float64))

    def rebuild_normalized(self):
        """Rebuild per-node normalized tensors + ts->pos maps from the raw
        buffers using the CURRENT scaler state. Called once per block (after the
        scalers update) — the forecaster also calls it when adopting a new
        snapshot. Columns: [V, ang, P, Q, sin_time, cos_time]."""
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
                sins[i] = s
                coss[i] = c
            mat = np.stack([Vn, An, Pn, Qn, sins, coss], axis=1)
            self.tensors[nid] = torch.tensor(mat, dtype=torch.float32)
            self.pos_by_ts[nid] = {int(ts): i for i, ts in enumerate(ts_col)}

    def _valid_positions(self, nid):
        """Row positions with a full HIST history behind and FUT targets ahead
        (i.e. positions that can form a complete training/forecast sample)."""
        T = self.tensors[nid].shape[0]
        return range(HIST, T - FUT)

    def build_training_indices(self):
        """All valid (nid, position, ts) samples across the retained buffer,
        subsampled to MAX_WINDOW_SAMPLES if exceeded (bounds train time/memory).
        """
        idx = []
        for nid in range(self.num_nodes):
            if nid not in self.tensors:
                continue
            pos_ts = {v: k for k, v in self.pos_by_ts[nid].items()}
            for t in self._valid_positions(nid):
                idx.append((nid, t, pos_ts[t]))
        if len(idx) > MAX_WINDOW_SAMPLES:
            chosen = np.random.choice(
                len(idx), MAX_WINDOW_SAMPLES, replace=False
            )
            idx = [idx[i] for i in chosen]
        return idx

    def build_forecast_indices(self, block_start, block_end):
        """Samples whose base timestamp is in [block_start, block_end) and whose
        FUT targets are all present in the buffer (so they can be scored)."""
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
# MODEL
# =====================================================
class DNN(nn.Module):
    """The forecasting network: a per-node embedding concatenated with the
    input feature vector, through a small MLP that outputs the FUT-step
    V/angle forecast. The node embedding lets one shared network specialize
    per node."""

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
    """Construct the model and its training companions on DEVICE.
    Returns (model, optimizer, scheduler, criterion, amp_scaler). The forecaster
    calls this too (for a shape-correct model to load snapshots into) and simply
    ignores the training-only return values."""
    model = DNN(num_nodes=num_nodes, dropout_p=DROPOUT_P).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.6
    )
    criterion = nn.MSELoss()
    scaler_amp = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    return model, optimizer, scheduler, criterion, scaler_amp
