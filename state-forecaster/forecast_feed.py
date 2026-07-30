
import os
import time
import json

from forecast_common import (utc_str, setup_logger, LOG_DIR, TS_INCREMENT_SEC,
                             DONE)

FEEDER_LOG = f"{LOG_DIR}/feeder.log"

#JSON_PATH = "results_data_forecasting_13_1min.jsonl"
JSON_PATH = "results_data_forecasting_13_5min.jsonl"
#JSON_PATH = "results_data_forecasting_13_15min.jsonl"
#JSON_PATH = "results_data_forecasting_123_5min.jsonl"

#FEED_RATE_HZ = 1.0          # records/sec the feeder emits (real ~1; >1 speeds testing)
#FEED_RATE_HZ = 4.0          # records/sec the feeder emits (real ~1; >1 speeds testing)
FEED_RATE_HZ = 50.0          # records/sec the feeder emits (real ~1; >1 speeds testing)

FEEDER_POLL_SEC = 0.05       # feeder idle poll interval

CADENCE_CHECK_SAMPLES = 20   # records to sample for the startup cadence check


def _pq_value(x):
    """Coerce a P or Q field to float. The State Estimator publishes the string
    "NA" for buses without a P/Q estimate (e.g. SOURCEBUS); map those to 0.0.
    None is also treated as 0.0."""
    if x == "NA" or x is None:
        return 0.0
    return float(x)


def parse_sv_entry(entry):
    """Parse one SvEstVoltages entry into (node_key, node_values).
    Single source of truth for entry interpretation, shared by the bus callback
    and the file reader. Node key = ConnectivityNode + "." + phase;
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
# FEEDER PROCESS
# Source is either the GridAPPS-D bus (gappsd_simid set) or a file (simid None).
# Both drivers funnel records through the shared per-record imputer step() and
# a shared emit() that enqueues the resulting burst to BOTH data queues.
# =====================================================
def feeder_proc(train_data_q, fc_data_q, sim_done, gappsd_simid):
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

    if gappsd_simid is not None:
        # ---------- BUS DRIVER (GridAPPS-D) ----------
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

    else:
        # ---------- FILE DRIVER ----------
        log.info(f"FEEDER start | path={JSON_PATH} | rate={FEED_RATE_HZ} Hz "
                 f"| increment={TS_INCREMENT_SEC}s")
        for record in read_json_records(JSON_PATH):
            emit(record, pace=True)

    sim_done.set()
    train_data_q.put(DONE)
    fc_data_q.put(DONE)
    log.info(f"FEEDER done | total={n_real} real + {n_imp} imputed "
             f"| last_ts={utc_str(last_ts) if last_ts else 'n/a'} | sent DONE")
    train_data_q.cancel_join_thread()
    fc_data_q.cancel_join_thread()

