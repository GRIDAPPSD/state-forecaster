
import io
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from forecast_common import (utc_str, setup_logger, LOG_DIR, DONE)
from forecast_dnn import (RollingBuffer, build_model, assemble_input_vector,
                          extract_scaler_state, HIST, FUT, USE_CURRENT_PQ,
                          DAY_LAG_SEC, WEEK_LAG_SEC, DEVICE)

TRAINER_LOG = f"{LOG_DIR}/trainer.log"

BLOCK_DAYS = 2           # training block size ("2-day window")
BLOCK_SEC = BLOCK_DAYS * 24 * 3600

EPOCHS_PER_BLOCK = 8
BATCH_SIZE = 512
NUM_WORKERS = 0
VAL_FRACTION = 0.05

PIN_MEMORY = (DEVICE == "cuda")

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
            X = X.to(DEVICE); y = y.to(DEVICE); nid = nid.long().to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
                loss = criterion(model(X, nid), y)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer); scaler_amp.update()
            total_loss += loss.item()
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for X, y, nid, _ in val_loader:
                X = X.to(DEVICE); y = y.to(DEVICE); nid = nid.long().to(DEVICE)
                val_loss += criterion(model(X, nid), y).item()
        scheduler.step()
        log(f"  Epoch {epoch+1:02d} | train={total_loss:.4f} | val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss; wait = 0
        else:
            wait += 1
            if wait >= patience:
                log("  Early stopping."); break


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
            train_data_q.cancel_join_thread()
            model_q.cancel_join_thread()
            return

        item = train_data_q.get()  # blocking; keep-all FIFO
        if isinstance(item, str) and item == DONE:
            model_q.put(DONE)
            log.info("TRAINER received DONE on queue → sent DONE to model queue → exit")
            train_data_q.cancel_join_thread()
            model_q.cancel_join_thread()
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
                log.info("TRAINER received DONE event mid-catchup "
                         "→ sent DONE to model queue → exit")
                train_data_q.cancel_join_thread()
                model_q.cancel_join_thread()
                return
            block_id += 1
            version += 1
            trainer_train_block(buf, model, optimizer, scheduler, criterion,
                                scaler_amp, block_start, block_end, block_id,
                                version, model_q, log)
            block_start = block_end
            block_end = block_start + BLOCK_SEC

        buf.append_record(record)

