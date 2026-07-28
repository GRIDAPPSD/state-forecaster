#!/usr/bin/python3
# =====================================================
# Change 3a — Piece 1: three-process scaffolding (STUBBED bodies).
# feeder → (data queues) → trainer & forecaster; trainer → (model queue) → forecaster.
# Verifies plumbing ONLY: paced feed, keep-all vs latest-only queues,
# model-snapshot handoff, model-queue-first priority, DONE shutdown,
# per-process log files. NO real training/forecasting yet (Pieces 2 & 3).
# =====================================================
import sys
import os
import io
import time
import queue
import logging
from collections import deque
from itertools import islice
import json  # for emitting forecast JSON

from forecaster_single import (
    read_json_records, make_imputer, RollingBuffer, build_model,
    train_block, extract_scaler_state, apply_scaler_state,
    assemble_input_vector, build_forecast_json, _pq_value,
    HIST, FUT, USE_CURRENT_PQ, TS_INCREMENT_SEC,
    DAY_LAG_SEC, WEEK_LAG_SEC, RETENTION_SEC, COMPUTE_LIVE_MAE,
    encode_phase, time_features, device,
    JSON_PATH, BLOCK_SEC, VAL_FRACTION, utc_str,
)
import numpy as np
import torch
import torch.multiprocessing as mp

# =====================================================
# CONFIG (3a additions)
# =====================================================
#FEED_RATE_HZ = 1.0          # records/sec the feeder emits (real ~1; >1 speeds testing)
#FEED_RATE_HZ = 4.0          # records/sec the feeder emits (real ~1; >1 speeds testing)
FEED_RATE_HZ = 20.0          # records/sec the feeder emits (real ~1; >1 speeds testing)
FORECASTER_POLL_SEC = 0.05   # forecaster idle poll interval when no work is pending
FEEDER_POLL_SEC = 0.05       # feeder idle poll interval
LOG_DIR = "."
TRAINER_LOG = f"{LOG_DIR}/trainer.log"
FORECASTER_LOG = f"{LOG_DIR}/forecaster.log"
FEEDER_LOG = f"{LOG_DIR}/feeder.log"

DONE = "__DONE__"           # sentinel Queue item meaning "end of stream"

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
            msgdict = message['message']
            ts = int(msgdict['timestamp'])
            sv = msgdict['Estimate']['SvEstVoltages']
            # build internal record: {"timestamp", "nodes": {key: {P,Q,V,Angle}}}
            nodes = {}
            for entry in sv:
                node_key = f"{entry['ConnectivityNode']}.{entry['phase']}"
                nodes[node_key] = {
                    "P": _pq_value(entry["P"]),
                    "Q": _pq_value(entry["Q"]),
                    "V": float(entry["vpu"]),
                    "Angle": float(entry["angleRad"]),
                }
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

    train_data_q.put(DONE)
    fc_data_q.put(DONE)
    log.info(f"FEEDER done | total={n_real} real + {n_imp} imputed "
             f"| last_ts={utc_str(last_ts) if last_ts else 'n/a'} | sent DONE")


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
