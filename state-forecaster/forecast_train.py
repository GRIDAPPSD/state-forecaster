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
import queue
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from forecast_common import utc_str, setup_logger, LOG_DIR
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

TRAINER_POLL_SEC = 0.1 # how often a blocked trainer wakes to re-check sim_done

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
    buf, model, optimizer, scheduler, criterion, scaler_amp,
    block_start, block_end, block_id, version, model_q, sim_done, log,  # sim_done added
):
    """Process one completed 2-day block: update scalers (causal), rebuild
    normalized tensors, train on the retained buffer, publish the trained
    snapshot (UNLESS the sim ended during this block's training — then the
    snapshot has no consumer, so skip it), and evict past the retention horizon."""
    buf.update_scalers_with_block(block_start, block_end)
    buf.rebuild_normalized()
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
        log.info(f"  block {block_id}: no training samples")
    else:
        train_block(
            model, optimizer, scheduler, criterion, scaler_amp,
            tr_idx, val_idx, buf, block_id, log=log.info,
        )
    # Publish the snapshot only if the sim is still running. If sim_done was set
    # while this block trained, the forecaster is shutting down and would never
    # consume it, so skip the (expensive) serialize + put.
    if sim_done.is_set():
        log.info(f"  block {block_id}: sim ended during training — snapshot v{version} not pushed")
    else:
        model_q.put(snapshot_to_bytes(model, buf, version))
        log.info(f"pushed model snapshot v{version}")
    buf.evict_old()


def trainer_proc(train_data_q, model_q, sim_done):
    """Trainer process entry point.

    Consumes records into its own RollingBuffer, training and publishing a model
    snapshot each time a 2-day block boundary is crossed. Shutdown is driven
    solely by the shared sim_done Event: once set, the trainer trains no
    further blocks and exits, abandoning any queued backlog.  A block already
    mid-training finishes (its push is suppressed — see trainer_train_block).

    Args:
        train_data_q: keep-all queue of records from the data feeder.
        model_q:      queue this process PUTs model snapshots on.
        sim_done:     shared Event; when set, stop and exit.
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
        # Shutdown: the sim is over -> train no further blocks, abandon backlog.
        if sim_done.is_set():
            log.info("TRAINER sim_done → exit (no further training)")
            train_data_q.cancel_join_thread()
            model_q.cancel_join_thread()
            return

        # Timeout get so a blocked trainer periodically wakes to re-check
        # sim_done
        try:
            record = train_data_q.get(timeout=TRAINER_POLL_SEC)
        except queue.Empty:
            continue

        ts = int(record["timestamp"])

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

        # Close out completed block(s). Check sim_done before each so we don't
        # train a pile of queued blocks at end-of-sim (no forecaster consumer).
        while ts >= block_end:
            if sim_done.is_set():
                log.info("TRAINER sim_done mid-catchup → exit")
                train_data_q.cancel_join_thread()
                model_q.cancel_join_thread()
                return
            block_id += 1
            version += 1
            trainer_train_block(
                buf, model, optimizer, scheduler, criterion, scaler_amp,
                block_start, block_end, block_id, version, model_q, sim_done, log,
            )
            block_start = block_end
            block_end = block_start + BLOCK_SEC

        buf.append_record(record)

