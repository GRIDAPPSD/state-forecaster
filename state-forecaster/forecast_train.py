"""
forecast_train.py — Trainer process for the State Forecaster.

Consumes the keep-all stream of state estimates, accumulates them into fixed
2-day training blocks, trains the DNN on each completed block, and publishes
the trained model (weights + scaler state) as a snapshot to the forecaster.

The trainer exists solely to keep the forecaster supplied with an up-to-date
model; it produces no forecasts itself. It stops promptly at end-of-simulation
(via the sim_done Event) rather than training blocks no forecaster will consume.

Contains: ReplayDataset / make_loader (turn buffer indices into training
batches), train_block (the epoch loop), snapshot_to_bytes (cross-process model
serialization), and trainer_proc (the process entry point).
"""

import io
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from forecast_common import utc_str, setup_logger, LOG_DIR, DONE
from forecast_dnn import (
    RollingBuffer,
    build_model,
    assemble_input_vector,
    extract_scaler_state,
    HIST,
    FUT,
    USE_CURRENT_PQ,
    DAY_LAG_SEC,
    WEEK_LAG_SEC,
    DEVICE,
)

TRAINER_LOG = f"{LOG_DIR}/trainer.log"

BLOCK_DAYS = 2  # training block size ("2-day window")
BLOCK_SEC = BLOCK_DAYS * 24 * 3600

EPOCHS_PER_BLOCK = 8  # max epochs per block (early stopping may cut short)
BATCH_SIZE = 512
NUM_WORKERS = 0  # DataLoader workers; 0 = load in the main process
VAL_FRACTION = 0.05  # fraction of a block's samples held out for validation

PIN_MEMORY = DEVICE == "cuda"  # pinned memory speeds host->GPU copies


# =====================================================
# DATASET (reads from a RollingBuffer snapshot)
# =====================================================
class ReplayDataset(Dataset):
    """PyTorch Dataset over a RollingBuffer's normalized tensors.

    Each index is a (node_id, position, timestamp) triple identifying one
    training sample; __getitem__ assembles that sample's input feature vector
    (recent history + current injection + day/week lag rows + phase + time
    features) and its FUT-step target (future V/angle)."""

    def __init__(self, indices, buf):
        self.indices = indices
        self.buf = buf

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        nid, t, ts = self.indices[idx]
        arr = self.buf.tensors[nid]
        phase = self.buf.phase[nid]

        # Position-based recent history: HIST rows before t, columns 0:2 (V, angle).
        hist_va = arr[t - HIST : t, 0:2].reshape(-1)
        # Current (or previous) injection P,Q at the base row (columns 2:4).
        base_pq = arr[t, 2:4] if USE_CURRENT_PQ else arr[t - 1, 2:4]

        # Day/week lag rows by TIMESTAMP lookup (not position): None if that
        # lagged timestamp isn't in the buffer -> assembler zero-flags it.
        day_pos = self.buf.pos_by_ts[nid].get(ts - DAY_LAG_SEC, None)
        week_pos = self.buf.pos_by_ts[nid].get(ts - WEEK_LAG_SEC, None)
        day_row = arr[day_pos] if day_pos is not None else None
        week_row = arr[week_pos] if week_pos is not None else None

        time_feat = arr[t, 4:6]  # sin/cos time-of-day features at the base row

        X = assemble_input_vector(
            hist_va, base_pq, day_row, week_row, phase, time_feat
        )
        # Target: the next FUT rows' V,angle (columns 0:2), flattened.
        y = arr[t + 1 : t + 1 + FUT, 0:2].reshape(-1)
        meta = {"nid": int(nid), "t": int(t), "Timestamp": int(ts)}
        return X, y, torch.tensor(nid, dtype=torch.long), meta


def make_loader(indices, buf, shuffle):
    """Wrap a ReplayDataset in a DataLoader with the module's batch settings."""
    ds = ReplayDataset(indices, buf)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )


# =====================================================
# TRAINING
# =====================================================
def train_block(
    model,
    optimizer,
    scheduler,
    criterion,
    scaler_amp,
    train_idx,
    val_idx,
    buf,
    block_id,
    log=print,
):
    """Train the model on one block's samples for up to EPOCHS_PER_BLOCK epochs,
    with early stopping (patience 2) on validation loss.

    Trains cumulatively on the passed-in model (weights carry over block to
    block). Uses AMP autocast + GradScaler on CUDA. Logs per-epoch train/val
    loss via `log` (the trainer passes log.info so it lands in trainer.log).
    """
    train_loader = make_loader(train_idx, buf, shuffle=True)
    val_loader = make_loader(val_idx, buf, shuffle=False)
    best_val = float("inf")
    patience = 2
    wait = 0
    for epoch in range(EPOCHS_PER_BLOCK):
        model.train()
        total_loss = 0.0
        for X, y, nid, _ in train_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            nid = nid.long().to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                loss = criterion(model(X, nid), y)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            total_loss += loss.item()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y, nid, _ in val_loader:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                nid = nid.long().to(DEVICE)
                val_loss += criterion(model(X, nid), y).item()
        scheduler.step()
        log(
            f"  Epoch {epoch+1:02d} | train={total_loss:.4f} | val={val_loss:.4f}"
        )
        # Early stopping: stop once val loss hasn't improved for `patience` epochs.
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                log("  Early stopping.")
                break


# =====================================================
# TRAINER PROCESS
# =====================================================
def snapshot_to_bytes(model, buf, version):
    """Serialize weights + scaler state to a bytes blob for cross-process
    transport. The scalers travel WITH the model so the forecaster normalizes
    inputs exactly as the trainer did.

    Bytes (via torch.save) rather than live tensors on the queue: a queued
    tensor's shared-memory backing is owned by the sender, so the receiver can
    fail to read it once the sender exits. A plain bytes blob has no such
    lifecycle dependency."""
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    buf_io = io.BytesIO()
    torch.save(
        {"weights": sd, "scaler_state": extract_scaler_state(buf)}, buf_io
    )
    return {"version": version, "blob": buf_io.getvalue()}


def trainer_train_block(
    buf,
    model,
    optimizer,
    scheduler,
    criterion,
    scaler_amp,
    block_start,
    block_end,
    block_id,
    version,
    model_q,
    log,
):
    """Process one completed 2-day block end to end: update the incremental
    scalers with this block's raw values (causal), rebuild the normalized
    tensors, train on the whole retained buffer, publish the trained snapshot,
    and evict rows past the retention horizon."""
    # 1) Causal scaler update using ONLY this block's raw values.
    buf.update_scalers_with_block(block_start, block_end)
    # 2) Rebuild normalized tensors from the raw buffers with updated scalers.
    buf.rebuild_normalized()
    # 3) Build the training set from the whole retained buffer (subsampled to
    #    MAX_WINDOW_SAMPLES inside build_training_indices), then split off val.
    train_idx = buf.build_training_indices()
    np.random.shuffle(train_idx)
    n_val = int(len(train_idx) * VAL_FRACTION)
    val_idx = train_idx[:n_val]
    tr_idx = train_idx[n_val:]
    log.info(
        f"[TRAIN] block {block_id} {utc_str(block_start)} → {utc_str(block_end)} "
        f"| train={len(tr_idx)} val={len(val_idx)}"
    )
    if len(tr_idx) == 0:
        # Too little data for any training sample (e.g. a very short run);
        # push current weights anyway so the forecaster still gets a snapshot.
        log.info(
            f"  block {block_id}: no training samples — pushing current weights"
        )
    else:
        # train_block logs epoch/loss via log.info -> lands in trainer.log.
        train_block(
            model,
            optimizer,
            scheduler,
            criterion,
            scaler_amp,
            tr_idx,
            val_idx,
            buf,
            block_id,
            log=log.info,
        )
    # 4) Publish the trained snapshot (weights + scaler state) to the forecaster.
    model_q.put(snapshot_to_bytes(model, buf, version))
    log.info(f"pushed model snapshot v{version}")
    # 5) Evict rows older than the retention horizon to bound memory.
    buf.evict_old()


def trainer_proc(train_data_q, model_q, sim_done):
    """Trainer process entry point.

    Consumes records from the keep-all data queue into its own RollingBuffer,
    training and publishing a model snapshot each time a 2-day block boundary is
    crossed. Exits promptly when the simulation ends — either the sim_done Event
    is set or a DONE sentinel arrives on the queue — training NO further blocks
    (once the sim is over there is no forecaster consumer for more snapshots).

    Args:
        train_data_q: keep-all queue of records from the data feeder.
        model_q:      queue this process PUTs model snapshots on (+ DONE at end).
        sim_done:     shared Event; when set, stop training and exit.
    """
    log = setup_logger("trainer", TRAINER_LOG)
    log.info("TRAINER start")
    buf = None
    model = optimizer = scheduler = criterion = scaler_amp = None
    block_start = None
    block_end = None
    block_id = 0
    version = 0

    while True:
        # End-of-sim check BEFORE blocking on the queue: stop immediately, train
        # nothing further, and forward DONE so the forecaster shuts down too.
        if sim_done.is_set():
            model_q.put(DONE)
            log.info(
                "TRAINER received DONE event → sent DONE to model queue → exit"
            )
            train_data_q.cancel_join_thread()
            model_q.cancel_join_thread()
            return

        item = train_data_q.get()  # blocking; keep-all FIFO
        # DONE on the data queue is a backstop for the sim_done Event; same exit.
        if isinstance(item, str) and item == DONE:
            model_q.put(DONE)
            log.info(
                "TRAINER received DONE on queue → sent DONE to model queue → exit"
            )
            train_data_q.cancel_join_thread()
            model_q.cancel_join_thread()
            return

        record = item
        ts = int(record["timestamp"])

        # Lazy init from the first record: discover the node set and build the
        # buffer + model/optimizer/etc. once, and anchor the first block window.
        if buf is None:
            node_names = sorted(record["nodes"].keys())
            buf = RollingBuffer(node_names)
            model, optimizer, scheduler, criterion, scaler_amp = build_model(
                buf.num_nodes
            )
            block_start = ts
            block_end = block_start + BLOCK_SEC
            log.info(
                f"lazy-init | {buf.num_nodes} nodes | first_ts={utc_str(ts)} "
                f"| block_end={utc_str(block_end)}"
            )

        # Close out any completed block(s) before ingesting this record. A
        # single far-future record can cross several boundaries, so this loops;
        # the sim_done check inside prevents training a pile of queued blocks at
        # end-of-sim (which would have no forecaster consumer).
        while ts >= block_end:
            if sim_done.is_set():
                model_q.put(DONE)
                log.info(
                    "TRAINER received DONE event mid-catchup "
                    "→ sent DONE to model queue → exit"
                )
                train_data_q.cancel_join_thread()
                model_q.cancel_join_thread()
                return
            block_id += 1
            version += 1
            trainer_train_block(
                buf,
                model,
                optimizer,
                scheduler,
                criterion,
                scaler_amp,
                block_start,
                block_end,
                block_id,
                version,
                model_q,
                log,
            )
            block_start = block_end
            block_end = block_start + BLOCK_SEC

        buf.append_record(record)
