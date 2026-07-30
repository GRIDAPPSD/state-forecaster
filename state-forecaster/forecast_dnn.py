import math
import numpy as np

from datetime import datetime, timezone
from collections import deque

import torch
import torch.nn as nn

DROPOUT_P = 0.03  # training regularization only (no MC dropout)
MAX_WINDOW_SAMPLES = 500_000  # cap: bounds per-block memory AND train time.

# --- history / horizon (in SAMPLES, i.e. timestamps) ---
HIST = 15  # past samples used as input (15 min @ 1-min)
FUT = 15  # future samples to forecast (15 min @ 1-min)

INPUT_DIM = (
    HIST * 2  # historical V, angle
    + 2  # current or previous P,Q
    + 2  # 1-day lag P,Q
    + 2  # 1-day lag V, angle
    + 1  # 1-day lag availability flag
    + 2  # 1-week lag P,Q
    + 2  # 1-week lag V, angle
    + 1  # 1-week lag availability flag
    + 3  # phase
    + 2  # sin_time, cos_time
)
OUTPUT_DIM = FUT * 2

# --- lag features (in TIME, converted to seconds) ---
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

USE_CURRENT_PQ = True

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

USE_GPU = True
DEVICE = "cuda" if torch.cuda.is_available() and USE_GPU else "cpu"


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


def assemble_input_vector(
    hist_va, base_pq, day_row, week_row, phase, time_feat
):
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
        "P": (buf.sc_P.min, buf.sc_P.max),
        "Q": (buf.sc_Q.min, buf.sc_Q.max),
        "V": (buf.sc_V.min, buf.sc_V.max),
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
        self.phase = {
            nid: torch.tensor(encode_phase(n), dtype=torch.float32)
            for nid, n in self.id_to_node.items()
        }
        # rebuilt each block:
        self.tensors = (
            {}
        )  # nid -> float32 [T,6]: V,ang,P,Q,sin,cos (normalized)
        self.pos_by_ts = {}  # nid -> {ts: row index}
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
        """Rebuild per-node normalized tensors + ts->pos maps from raw buffers,
        using the CURRENT scaler state. Called once per block after scalers update.
        """
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
            chosen = np.random.choice(
                len(idx), MAX_WINDOW_SAMPLES, replace=False
            )
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
    model = DNN(num_nodes=num_nodes, dropout_p=DROPOUT_P).to(DEVICE)
    # --- FUTURE HOOK (Change 3): model.share_memory() before spawning the
    #     forecast process so weight updates propagate without files. ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.6
    )
    criterion = nn.MSELoss()
    scaler_amp = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    return model, optimizer, scheduler, criterion, scaler_amp
