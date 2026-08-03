"""
forecast_predict.py — Forecaster process for the State Forecaster.

Consumes the keep-all stream of state estimates into its own RollingBuffer,
adopts model snapshots as the trainer publishes them, and — once it has a
model — produces a forecast for each newest real estimate: the FUT-step-ahead
V/angle for every node. Forecasts are published to the GridAPPS-D bus (bus mode)
and/or written to a .jsonl file, and optionally scored after the fact against
the actual estimates that later arrive (deferred MAE).

Design notes:
  * Ingests EVERY record (real + imputed) to keep a gapless recent-history
    window, but only forecasts the LATEST real timestamp (never stale ones).
  * Incremental normalization (NormStore): normalize one row per arriving
    estimate; the expensive full re-normalize happens only when a new snapshot's
    scalers arrive (rare) — so per-estimate cost stays low and scales to many
    nodes.
  * Deferred MAE: a forecast can't be scored when made (its future hasn't
    happened yet), so one forecast per model version is held and scored as its
    target timestamps' actual estimates arrive.

Contents: build_forecast_json (output message), drain_latest/drain_all (queue
helpers), normalize_row + NormStore (incremental normalized history),
forecast_latest (build inputs + run the model), forecaster_proc (entry point).
"""

import os
import io
import time
import queue
import json
import numpy as np
from collections import deque
from itertools import islice

import torch

from forecast_common import (
    utc_str,
    setup_logger,
    LOG_DIR,
    TS_INCREMENT_SEC,
    DONE,
)
from forecast_dnn import (
    RollingBuffer,
    build_model,
    assemble_input_vector,
    apply_scaler_state,
    time_features,
    HIST,
    FUT,
    USE_CURRENT_PQ,
    RETENTION_SEC,
    DAY_LAG_SEC,
    WEEK_LAG_SEC,
    DEVICE,
)

FORECASTER_LOG = f"{LOG_DIR}/forecaster.log"

# Each published forecast is appended here (one JSON per line) for offline
# validation/plotting. None disables. Large on long runs (one forecast per
# estimate); intended to be suppressible for production/speed.
FORECAST_OUTPUT_JSONL = f"{LOG_DIR}/forecast_output.jsonl"

FORECASTER_POLL_SEC = 0.05  # idle poll interval when no work is pending

# Deferred scoring: score one forecast per model version against the actual
# estimates that later arrive. Off for production-speed runs.
COMPUTE_LIVE_MAE = True

# Per-step MAE at specific horizon steps (1-based indices into the FUT-step
# forecast), to see how accuracy degrades further out. None disables. Each
# listed step gets its own V/angle MAE on a separate [MAE-STEPS] log line.
COMPUTE_MAE_STEPS = [1, 8, 15]

# Log 1 of every N forecasts (a heartbeat; avoids flooding the log since a
# forecast is producedoming estimate).
FORECAST_LOG_EVERY = 200


def build_forecast_json(
    preds, nids, base_ts, buf, base_time, simulation_id=None
):
    """Assemble the published forecast message for ONE base timestamp.
    Physical units (per-unit V, radian angle); epoch-seconds timestamps.

    Node identity: internally the NN uses a single combined key (e.g. "632.1").
    On output we split it back on the LAST dot into separate ConnectivityNode
    ("632") and phase ("1") fields — matching the GridAPPS-D publish
    convention. Phase is emitted AS-IS (no mapping); dots are only separators.

    Message shape: top level carries generic ADMS fields (timestamp,
    simulation_id); the forecast-specific payload is nested under "Forecast"
    (parallel to how the State Estimator nests its data under "Estimate").
    """
    row_mask = base_ts == base_time
    sel_preds = preds[row_mask]
    sel_nids = nids[row_mask]

    # Nominal future timestamps: base + k increments, for k = 1..FUT.
    forecast_times = [
        int(base_time + TS_INCREMENT_SEC * (k + 1)) for k in range(FUT)
    ]

    nodes_out = {}
    for row, nid in zip(sel_preds, sel_nids):
        node_key = buf.id_to_node[int(nid)]
        if "." in node_key:
            cn, phase = node_key.rsplit(".", 1)
        else:
            cn, phase = node_key, ""
        # Model output interleaves [V, ang, V, ang, ...] across the horizon;
        # de-interleave and inverse-transform each back to physical units.
        V_series = buf.sc_V.inverse(row[0::2])
        ang_series = buf.sc_ang.inverse(row[1::2])
        nodes_out[node_key] = {
            "ConnectivityNode": cn,
            "phase": phase,
            "V": [float(v) for v in V_series],
            "Angle": [float(a) for a in ang_series],
        }

    return {
        "timestamp": int(base_time),  # top-level: generic ADMS field
        "simulation_id": simulation_id,  # top-level: generic ADMS field
        "Forecast": {  # forecast-specific payload nested here
            "step_sec": TS_INCREMENT_SEC,
            "horizon": FUT,
            "forecast_times": forecast_times,
            "nodes": nodes_out,
        },
    }


# =====================================================
# QUEUE HELPERS
# =====================================================
def drain_latest(q):
    """Consumer-side drain returning (latest_non_DONE_item, saw_done).
    Keeps only the newest real item (older ones discarded); reports whether a
    DONE sentinel was seen (never lost, even alongside a real item). Used for
    the model queue, where only the newest snapshot matters. Non-blocking;
    returns (None, False) if the queue was empty."""
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
    """Drain ALL pending items in order, returning (records_list, saw_done).
    Used for the forecaster's keep-all data queue so recent history stays
    gapless (every record is kept, unlike drain_latest). Non-blocking."""
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
# INCREMENTAL NORMALIZED HISTORY (forecaster side)
# =====================================================
def normalize_row(scalers, V, ang, P, Q, ts):
    """Normalize one raw row into [V,ang,P,Q,sin,cos] float32 using the given
    scalers tuple (sc_V, sc_ang, sc_P, sc_Q). Mirrors how RollingBuffer builds
    its normalized tensors, so the forecaster's rows match the trainer's."""
    sc_V, sc_ang, sc_P, sc_Q = scalers
    Vn = float(sc_V.transform(np.array([V], dtype=np.float64))[0])
    an = float(sc_ang.transform(np.array([ang], dtype=np.float64))[0])
    Pn = float(sc_P.transform(np.array([P], dtype=np.float64))[0])
    Qn = float(sc_Q.transform(np.array([Q], dtype=np.float64))[0])
    s, c = time_features(int(ts))
    return np.array([Vn, an, Pn, Qn, s, c], dtype=np.float32)


class NormStore:
    """Forecaster-side normalized history, maintained incrementally.

    Per node: an ordered deque of (ts, norm_row) for the position-based HIST
    window, plus a {ts: norm_row} dict for O(1) day/week lag lookups. Rows are
    normalized once on arrival (append) and reused; a full rebuild happens only
    when a new snapshot changes the scalers (rebuild_from_raw) — keeping the
    per-estimate cost low even at large node counts."""

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.rows = {nid: deque() for nid in range(num_nodes)}  # (ts, norm_row)
        self.by_ts = {nid: {} for nid in range(num_nodes)}  # ts -> norm_row

    def append(self, nid, ts, norm_row):
        self.rows[nid].append((ts, norm_row))
        self.by_ts[nid][ts] = norm_row

    def evict_before(self, cutoff_ts):
        """Drop rows older than cutoff_ts (keeps the store bounded, in step
        with the RollingBuffer's raw eviction)."""
        for nid in range(self.num_nodes):
            dq = self.rows[nid]
            bt = self.by_ts[nid]
            while dq and dq[0][0] < cutoff_ts:
                old_ts, _ = dq.popleft()
                bt.pop(old_ts, None)

    def rebuild_from_raw(self, buf, scalers):
        """Full re-normalize of everything currently in buf.raw with the given
        (newly adopted) scalers. Called once per snapshot arrival (rare), since
        changing scalers invalidates all previously-normalized rows."""
        for nid in range(self.num_nodes):
            self.rows[nid].clear()
            self.by_ts[nid].clear()
            for ts, V, ang, P, Q in buf.raw[nid]:
                nr = normalize_row(scalers, V, ang, P, Q, ts)
                self.rows[nid].append((ts, nr))
                self.by_ts[nid][ts] = nr


def forecast_latest(model, store, buf, latest_ts, log):
    """Produce a forecast for base timestamp latest_ts across all nodes.

    For each node with enough history, assemble its input vector (identical
    layout to training, via assemble_input_vector) from the last HIST+1
    normalized rows plus the day/week lag rows, then run all nodes through the
    model in one batch. Returns (preds, nids, base_ts_arr) for
    build_forecast_json, or None if no node has enough history yet.

    A node is skipped if its newest row isn't exactly at latest_ts (a gap at the
    front means we can't form a valid, aligned history window for it)."""
    X_list, nid_list = [], []

    for nid in range(store.num_nodes):
        dq = store.rows[nid]
        if len(dq) < HIST + 1:
            continue  # not enough consecutive history yet

        # Last HIST+1 rows: the final one is the base (latest_ts); the prior
        # HIST are the history window.
        recent = list(islice(reversed(dq), 0, HIST + 1))  # newest-first
        recent.reverse()  # oldest-first
        base_ts, base_row = recent[-1]
        if base_ts != latest_ts:
            continue  # node's newest row isn't at latest_ts (gap) — skip it

        hist_rows = recent[:HIST]  # the HIST rows before the base
        hist_va = torch.tensor(
            np.concatenate([r[0:2] for (_, r) in hist_rows]),
            dtype=torch.float32,
        )

        base_np = base_row if USE_CURRENT_PQ else recent[-2][1]
        base_pq = torch.tensor(base_np[2:4], dtype=torch.float32)

        # Day/week lag rows by timestamp lookup (None -> assembler zero-flags).
        day_np = store.by_ts[nid].get(latest_ts - DAY_LAG_SEC, None)
        week_np = store.by_ts[nid].get(latest_ts - WEEK_LAG_SEC, None)
        day_row = (
            torch.tensor(day_np, dtype=torch.float32)
            if day_np is not None
            else None
        )
        week_row = (
            torch.tensor(week_np, dtype=torch.float32)
            if week_np is not None
            else None
        )

        time_feat = torch.tensor(base_row[4:6], dtype=torch.float32)
        phase = buf.phase[nid]

        X = assemble_input_vector(
            hist_va, base_pq, day_row, week_row, phase, time_feat
        )
        X_list.append(X)
        nid_list.append(nid)

    if not X_list:
        return None

    # One batched forward pass over all eligible nodes.
    X_batch = torch.stack(X_list).to(DEVICE)
    nid_batch = torch.tensor(nid_list, dtype=torch.long, device=DEVICE)
    model.eval()
    with torch.no_grad():
        preds = model(X_batch, nid_batch).cpu().numpy()

    nids = np.array(nid_list)
    base_ts_arr = np.full(len(nid_list), latest_ts, dtype=np.int64)
    return preds, nids, base_ts_arr


# =====================================================
# FORECASTER PROCESS
# =====================================================
def forecaster_proc(fc_data_q, model_q, gappsd_simid):
    """Forecaster process entry point.

    Loop: adopt the newest model snapshot if any (model queue first), ingest all
    newly-arrived estimates (keeping history gapless), forecast the latest real
    timestamp, publish/record it, and — if enabled — score a held forecast per
    model version. Exits once end-of-stream has been seen on BOTH the data queue
    and the model queue (each latched via a sticky flag).

    Args:
        fc_data_q:    keep-all queue of records from the data feeder.
        model_q:      queue of model snapshots from the trainer (+ DONE at end).
        gappsd_simid: GridAPPS-D sim id -> publish to the bus; None -> file only.
    """
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

        os.environ["GRIDAPPSD_APPLICATION_ID"] = "state-forecaster"
        os.environ["GRIDAPPSD_APPLICATION_STATUS"] = "STARTED"
        os.environ["GRIDAPPSD_USER"] = "app_user"
        os.environ["GRIDAPPSD_PASSWORD"] = "1234App"
        gapps = GridAPPSD(gappsd_simid)
        assert gapps.connected
        log.info(
            f"FORECASTER connected to GridAPPS-D simid={gappsd_simid} "
            f"for publishing forecasts"
        )
        publish_to_topic = service_output_topic(
            "state-forecaster", gappsd_simid
        )

    buf = None
    model = None
    store = None  # NormStore (incremental normalized rows)
    scalers = None  # (sc_V, sc_ang, sc_P, sc_Q) — refs into buf's scalers
    have_scalers = False  # True once first snapshot applied
    current_version = 0
    pending_snap = None  # snapshot that arrived before the first data record
    data_done = False  # sticky: DONE seen on the data queue
    model_done = False  # sticky: DONE seen on the model queue
    fc_count = 0
    warmup_logged = False

    # --- deferred live MAE scoring state ---
    score_armed = False  # set on snapshot adoption; scores the NEXT forecast
    pending = (
        None  # the one forecast currently being scored (see score_pending)
    )

    if COMPUTE_MAE_STEPS is not None:
        assert all(
            1 <= s <= FUT for s in COMPUTE_MAE_STEPS
        ), f"COMPUTE_MAE_STEPS {COMPUTE_MAE_STEPS} has values outside 1..FUT={FUT}"

    # Forecast-output file: truncate/create empty at startup, then append one
    # JSON line per published forecast. Open/append/close per write (below) is
    # durable and needs no close at exit. None disables it.
    if FORECAST_OUTPUT_JSONL:
        with open(FORECAST_OUTPUT_JSONL, "w"):
            pass  # create/truncate to empty
        log.info(f"Writing published forecasts to {FORECAST_OUTPUT_JSONL}")

    def score_pending(record):
        """Deferred MAE: if this real estimate's timestamp matches a step of the
        pending (held) forecast, accumulate absolute error over the nodes present
        in both. Also accumulates per-step error for the steps in
        COMPUTE_MAE_STEPS. When all horizon steps have been collected, log the
        aggregate MAE (and the per-step line) and clear the pending forecast."""
        nonlocal pending
        if pending is None:
            return
        ts = int(record["timestamp"])
        if ts not in pending["remaining"]:
            return
        pred_at = pending["pred"][ts]
        step = pending["ts_to_step"][ts]  # 1-based horizon step of this ts
        track_step = COMPUTE_MAE_STEPS is not None and step in pending["step_v"]
        for node, vals in record["nodes"].items():
            p = pred_at.get(node)
            if p is None:
                continue  # node not in the forecast — skip (defensive)
            dv = abs(p[0] - vals["V"])
            da = abs(p[1] - vals["Angle"])
            pending["abs_v"] += dv
            pending["abs_a"] += da
            pending["n"] += 1
            if track_step:  # bucket into this step's per-step accumulators
                pending["step_v"][step] += dv
                pending["step_a"][step] += da
                pending["step_n"][step] += 1
        pending["remaining"].discard(ts)
        if not pending["remaining"]:  # all horizon steps collected -> report
            n = max(pending["n"], 1)
            log.info(
                f"[MAE] v{pending['version']} base_time="
                f"{utc_str(pending['base_time'])} | "
                f"Voltage MAE (pu): {pending['abs_v']/n:.6f} | "
                f"Angle MAE (rad): {pending['abs_a']/n:.6f} "
                f"({pending['n']} node-steps)"
            )
            if COMPUTE_MAE_STEPS is not None:  # per-step degradation line
                parts = []
                for s in COMPUTE_MAE_STEPS:
                    sn = max(pending["step_n"][s], 1)
                    parts.append(
                        f"step {s}: V={pending['step_v'][s]/sn:.6f} "
                        f"A={pending['step_a'][s]/sn:.6f}"
                    )
                log.info(
                    f"[MAE-STEPS] v{pending['version']} base_time="
                    f"{utc_str(pending['base_time'])} | " + " | ".join(parts)
                )
            pending = None

    def ingest(record):
        """Append the raw record to buf; if scalers are known, also normalize
        this row incrementally into the NormStore (per-node)."""
        ts = int(record["timestamp"])
        buf.append_record(record)  # updates buf.raw + buf.newest_ts
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
        """Evict raw and normalized rows past the retention horizon (keeps both
        the RollingBuffer and the NormStore bounded)."""
        if buf.newest_ts is None:
            return
        cutoff = buf.newest_ts - RETENTION_SEC
        buf.evict_old()  # raw
        if store is not None:
            store.evict_before(cutoff)  # normalized

    while True:
        # 1) MODEL QUEUE FIRST: adopt newest snapshot (weights + scalers), and
        #    latch model_done if the trainer's DONE has arrived.
        snap, model_saw_done = drain_latest(model_q)
        if model_saw_done:
            model_done = True  # sticky, like data_done
        if snap is not None:
            if buf is None:
                # Snapshot arrived before any data; stash and apply after init.
                pending_snap = snap
            else:
                blob = torch.load(io.BytesIO(snap["blob"]), map_location="cpu")
                model.load_state_dict(blob["weights"])
                apply_scaler_state(buf, blob["scaler_state"])
                store.rebuild_from_raw(buf, scalers)  # full re-normalize (rare)
                have_scalers = True
                current_version = snap["version"]
                if COMPUTE_LIVE_MAE:
                    score_armed = True  # score the next forecast (this version)
                log.info(
                    f"adopted snapshot v{current_version} "
                    f"→ scalers updated, store re-normalized "
                    f"({sum(len(store.rows[n]) for n in range(store.num_nodes))} rows)"
                )

        # 2) DATA QUEUE: drain ALL (keep-all, gapless), ingest in order.
        records, data_saw_done = drain_all(fc_data_q)
        if data_saw_done:
            data_done = True

        latest_ts = None
        latest_imputed = False
        for record in records:
            if buf is None:
                # Lazy init from the first record (always REAL — no leading
                # imputation is ever emitted). Discover the node set, build the
                # buffer/model/store, and apply any snapshot that arrived early.
                node_names = sorted(record["nodes"].keys())
                buf = RollingBuffer(node_names)
                model, *_ = build_model(buf.num_nodes)
                store = NormStore(buf.num_nodes)
                scalers = (buf.sc_V, buf.sc_ang, buf.sc_P, buf.sc_Q)
                log.info(
                    f"lazy-init | {buf.num_nodes} nodes | first_ts={utc_str(int(record['timestamp']))}"
                )
                if pending_snap is not None:
                    blob = torch.load(
                        io.BytesIO(pending_snap["blob"]), map_location="cpu"
                    )
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
                score_pending(record)  # deferred MAE on real estimates only

        if records:
            evict()

        # 3) FORECAST the latest timestamp — but ONLY if it is a REAL estimate.
        # Imputed (interpolated) data is ingested for gapless history but must
        # not trigger a published forecast. Because the feeder emits each gap as
        # a [imp, ..., real] burst, the latest drained record is normally real;
        # if a drain lands mid-burst on an imputed record, we skip this cycle and
        # forecast on the real record that arrives next cycle.
        if latest_ts is not None and not latest_imputed:
            if have_scalers:
                result = forecast_latest(model, store, buf, latest_ts, log)
                if result is not None:
                    preds, nids, base_ts_arr = result
                    fc_count += 1
                    fc_json = build_forecast_json(
                        preds,
                        nids,
                        base_ts_arr,
                        buf,
                        base_time=latest_ts,
                        simulation_id=gappsd_simid,
                    )

                    # Publish to the bus (if connected) and/or record to file.
                    if gapps is not None:
                        gapps.send(publish_to_topic, json.dumps(fc_json))
                    if FORECAST_OUTPUT_JSONL:
                        with open(FORECAST_OUTPUT_JSONL, "a") as f:
                            f.write(json.dumps(fc_json) + "\n")

                    if fc_count == 1 or fc_count % FORECAST_LOG_EVERY == 0:
                        log.info(
                            f"[FORECAST] #{fc_count} base_time={utc_str(latest_ts)} "
                            f"using model v{current_version} | {len(fc_json['Forecast']['nodes'])} nodes"
                        )

                    # Arm deferred scoring: stash this forecast's per-node
                    # predictions keyed by their future timestamps, to be scored
                    # as those actual estimates arrive (once per model version).
                    if COMPUTE_LIVE_MAE and score_armed:
                        pred = {}
                        ts_to_step = {}  # forecast_time -> 1-based step index
                        for k in range(FUT):
                            ft = fc_json["Forecast"]["forecast_times"][k]
                            pred[ft] = {
                                node: (nd["V"][k], nd["Angle"][k])
                                for node, nd in fc_json["Forecast"][
                                    "nodes"
                                ].items()
                            }
                            ts_to_step[ft] = k + 1  # 1-based
                        pending = {
                            "remaining": set(pred.keys()),
                            "pred": pred,
                            "abs_v": 0.0,
                            "abs_a": 0.0,
                            "n": 0,
                            "version": current_version,
                            "base_time": latest_ts,
                            "ts_to_step": ts_to_step,
                            # per-step accumulators for the requested steps:
                            "step_v": {
                                s: 0.0 for s in (COMPUTE_MAE_STEPS or [])
                            },
                            "step_a": {
                                s: 0.0 for s in (COMPUTE_MAE_STEPS or [])
                            },
                            "step_n": {s: 0 for s in (COMPUTE_MAE_STEPS or [])},
                        }
                        score_armed = False
                        log.info(
                            f"[MAE] armed: scoring forecast v{current_version} "
                            f"base_time={utc_str(latest_ts)} over {FUT} steps"
                        )

            else:
                if not warmup_logged:
                    log.info(
                        f"latest_ts={utc_str(latest_ts)} | no model yet "
                        f"(warm-up) — skipping forecasts until first snapshot"
                    )
                    warmup_logged = True

        # 4) Shutdown once end-of-stream is seen on BOTH queues (sticky flags).
        # The two DONEs arrive at different times (feeder finishes well before
        # the trainer's final snapshot), so each is latched independently.
        if data_done and model_done:
            log.info(
                f"FORECASTER received DONE on both queues → exit "
                f"(total forecasts: {fc_count}, final model v{current_version})"
            )
            # Signal end-of-forecasts to our own downstream consumers (symmetry
            # with the State Estimator's COMPLETE message).
            if gapps is not None:
                done_json = {
                    "simulation_id": gappsd_simid,
                    "processStatus": "COMPLETE",
                }
                gapps.send(publish_to_topic, json.dumps(done_json))

            fc_data_q.cancel_join_thread()
            model_q.cancel_join_thread()
            return

        # 5) Avoid busy-spin when there was nothing to do this cycle.
        if not records and snap is None:
            time.sleep(FORECASTER_POLL_SEC)
