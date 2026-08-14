"""
forecast_feed.py — Data-feeder process for the State Forecaster.

Torch-free: this module knows nothing about neural networks. Its sole job is to
be the single data source for the app — read state estimates from either the
GridAPPS-D message bus or a .jsonl file, fill any timestamp gaps by
interpolation, and hand each record to the trainer and forecaster processes via
their queues.

Key pieces:
  * Parse helpers (_pq_value, parse_sv_entry, sv_entries_to_nodes,
    read_json_records) — the single source of truth for turning a State
    Estimator "SvEstVoltages" entry into the app's internal record shape.
  * make_imputer — a stateful per-record step that backfills missing
    grid-aligned timestamps and performs a one-time cadence sanity check.
  * feeder_proc — the process entry point; selects bus vs. file source and
    funnels every record through a shared emit() to both data queues.
"""

import os
import time
import json

from forecast_common import (
    utc_str,
    setup_logger,
    LOG_DIR,
    TS_INCREMENT_SEC,
)

FEEDER_LOG = f"{LOG_DIR}/data_feeder.log"

# Input file for FILE-DRIVER mode (used when no GridAPPS-D sim id is given).
# Uncomment the one matching the model / increment being tested;UST match TS_INCREMENT_SEC (the startup cadence check will complain if not).
# JSON_PATH = "results_data_forecasting_13_1min.jsonl"
# JSON_PATH = "results_data_forecasting_13_5min.jsonl"
JSON_PATH = "results_data_forecasting_13_4weeks.jsonl"
# JSON_PATH = "results_data_forecasting_13_15min.jsonl"
# JSON_PATH = "results_data_forecasting_123_5min.jsonl"

# FILE-DRIVER pacing: records/sec the feeder emits. Real estimates arrive ~1/s;
# rates >1 just speed up file-based testing. (Bus driver is not paced — it
# emits as messages arrive.) Note: very high rates make data arrive faster than
# training can consume it, so a run may finish before much forecasting occurs.
# FEED_RATE_HZ = 1.0
# FEED_RATE_HZ = 4.0
FEED_RATE_HZ = 10.0
# FEED_RATE_HZ = 50.0

FEEDER_POLL_SEC = 0.05  # bus-driver idle sleep while waiting for messages

# Number of initial real records whose spacing is sampled to infer the true
# data cadence for the startup cadence check (see make_imputer).
CADENCE_CHECK_SAMPLES = 20


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
    (per-unit), angleRad -> Angle (radians); P/Q "NA" -> 0.0. V/Angle are NOT
    NA-coerced (a missing voltage/angle should fail loudly, not be zeroed)."""
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
    """Build the internal {node_key: {P,Q,V,Angle}} dict from a SvEstVoltages
    list (applies parse_sv_entry to each entry)."""
    nodes = {}
    for entry in sv_entries:
        node_key, node_vals = parse_sv_entry(entry)
        nodes[node_key] = node_vals
    return nodes


def read_json_records(path):
    """Yield one internal record per line from the SE-written .jsonl file.
    File envelope per line: {"timeStamp": ..., "SvEstVoltages": [...]}.
    Yields the app's internal shape: {"timestamp", "nodes": {...}}."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ts = int(rec["timeStamp"])
            nodes = sv_entries_to_nodes(rec["SvEstVoltages"])  # shared
            yield {"timestamp": ts, "nodes": nodes}


def _emit_cadence_warning(min_delta, increment_sec):
    """Loud, hard-to-miss warning when the detected data cadence doesn't match
    the configured TS_INCREMENT_SEC. Printed once, to stdout (the shared
    console in multi-process), so it surfaces prominently rather than being
    buried in a log file."""
    bar = "*" * 72
    print("\n" + bar)
    print(
        "!!!!!!!!!!!!!!!!!!!!!!  CADENCE MISMATCH WARNING  !!!!!!!!!!!!!!!!!!!!!!"
    )
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
    # Cadence-check state: track the minimum spacing seen over the first several
    # real records; the smallest gap is the true cadence (gaps only enlarge it).
    cad_min_delta = None
    cad_count = 0
    cad_warned = False
    cad_mismatch = False  # True if cadence mismatched -> suppress per-gap spam

    def step(record):
        nonlocal prev_ts, prev_nodes
        nonlocal cad_min_delta, cad_count, cad_warned, cad_mismatch

        ts = int(record["timestamp"])
        out = []

        if prev_ts is not None:
            delta = ts - prev_ts

            # --- one-time cadence check on raw spacing (before imputation) ---
            if not cad_warned and delta > 0:
                cad_min_delta = (
                    delta
                    if cad_min_delta is None
                    else min(cad_min_delta, delta)
                )
                cad_count += 1
                # Conclude early on definitive evidence (a gap smaller than the
                # configured increment can't be explained away), else once we've
                # sampled enough records to trust the minimum as the true cadence.
                if (
                    cad_min_delta < increment_sec
                    or cad_count >= CADENCE_CHECK_SAMPLES
                ):
                    if cad_min_delta != increment_sec:
                        _emit_cadence_warning(cad_min_delta, increment_sec)
                        cad_mismatch = True
                    cad_warned = True

            if delta <= 0:
                # Non-increasing timestamps violate the stream's assumptions;
                # pass the record through unchanged rather than impute backward.
                print(
                    f"[IMPUTE] WARNING: non-increasing timestamp "
                    f"{prev_ts} -> {ts}; passing through without imputation."
                )
            elif delta % increment_sec != 0:
                # Gap isn't a whole number of increments -> can't place imputed
                # records on the grid. Warn once per gap, unless we've already
                # reported a cadence mismatch (which would spam every record).
                if not cad_mismatch:
                    print(
                        f"[IMPUTE] WARNING: gap {delta}s not a multiple of "
                        f"increment {increment_sec}s ({prev_ts} -> {ts}); "
                        f"no imputation for this gap."
                    )
            else:
                # Normal case: gap is N increments. If N > 1, synthesize the
                # N-1 missing grid points by linear interpolation per node.
                gap_steps = delta // increment_sec
                if gap_steps > 1:
                    for k in range(1, gap_steps):
                        frac = k / gap_steps
                        imp_ts = prev_ts + k * increment_sec
                        imp_nodes = {}
                        for node_name, cur_vals in record["nodes"].items():
                            pv = prev_nodes.get(node_name)
                            if pv is None:
                                # Node absent in the previous record: can't
                                # interpolate, so copy current values as-is.
                                imp_nodes[node_name] = dict(cur_vals)
                            else:
                                imp_nodes[node_name] = {
                                    "P": pv["P"]
                                    + frac * (cur_vals["P"] - pv["P"]),
                                    "Q": pv["Q"]
                                    + frac * (cur_vals["Q"] - pv["Q"]),
                                    "V": pv["V"]
                                    + frac * (cur_vals["V"] - pv["V"]),
                                    "Angle": pv["Angle"]
                                    + frac * (cur_vals["Angle"] - pv["Angle"]),
                                }
                        out.append(
                            {
                                "timestamp": imp_ts,
                                "nodes": imp_nodes,
                                "_imputed": True,
                            }
                        )

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
    """Data-feeder process entry point.

    Reads state estimates from the GridAPPS-D bus (if gappsd_simid is given) or
    from a .jsonl file (otherwise), runs each through the imputer, and enqueues
    the results to the trainer and forecaster data queues. On end-of-stream,
    sets the sim_done Event and releases the queues' background threads so
    the process can exit cleanly.

    Args:
        train_data_q: queue to the trainer (keep-all).
        fc_data_q:    queue to the forecaster (keep-all).
        sim_done:     shared Event set at end-of-stream.
        gappsd_simid: GridAPPS-D simulation id -> bus mode; None -> file mode.
    """
    log = setup_logger("data_feeder", FEEDER_LOG)

    # Per-record imputer step + cadence-guard (shared by whichever driver runs).
    step = make_imputer(TS_INCREMENT_SEC)

    # Emitted-record counters (shared across emit() calls via closure).
    n_real = 0
    n_imp = 0
    last_ts = None

    def emit(record, pace=False):
        """Run one record through the imputer and enqueue the resulting burst
        (imputed records + the real one) onto both queues. Imputed records go
        out back-to-back; optional pacing sleeps once, after the real record
        (file driver only — simulates realistic arrival spacing).
        """
        nonlocal n_real, n_imp, last_ts
        for out_rec in step(record):
            train_data_q.put(out_rec)
            fc_data_q.put(out_rec)
            if out_rec.get("_imputed", False):
                n_imp += 1
            else:
                n_real += 1
                last_ts = int(out_rec["timestamp"])
                if (
                    n_real % 60 == 0
                ):  # progress heartbeat for every 60 real records
                    log.info(
                        f"fed {n_real} real (+{n_imp} imputed) "
                        f"| latest_ts={utc_str(last_ts)}"
                    )
        if pace and FEED_RATE_HZ > 0:
            # Pace only on the real estimate (imputed records in a burst go out
            # immediately); throttles the file driver to a realistic rate.
            time.sleep(1.0 / FEED_RATE_HZ)

    if gappsd_simid is not None:
        # ---------- BUS DRIVER (GridAPPS-D) ----------
        # Subscribe to the State Estimator's output; each arriving message is
        # unwrapped and pushed through emit(). The process stays alive in a
        # poll loop until a processStatus=COMPLETE message ends the stream.
        from gridappsd import GridAPPSD
        from gridappsd.topics import service_output_topic

        os.environ["GRIDAPPSD_APPLICATION_ID"] = "state-forecaster"
        os.environ["GRIDAPPSD_APPLICATION_STATUS"] = "STARTED"
        os.environ["GRIDAPPSD_USER"] = "app_user"
        os.environ["GRIDAPPSD_PASSWORD"] = "1234App"

        keepLoopingFlag = True

        def estimateCallback(header, message):
            """Bus subscription callback (fires per incoming message on the
            gridappsd listener thread). Handles the COMPLETE end-of-stream
            signal, or unwraps an estimate message and emits its record."""
            nonlocal keepLoopingFlag
            if "processStatus" in message:
                if message["processStatus"] == "COMPLETE":
                    log.info("Got processStatus COMPLETE message")
                    keepLoopingFlag = False
                return
            # Unwrap the 3-level envelope: message -> message -> Estimate,
            # then take timeStamp + SvEstVoltages from the Estimate structure
            # (the same structure the SE writes to the .jsonl file).
            est = message["message"]["Estimate"]
            ts = int(est["timeStamp"])
            nodes = sv_entries_to_nodes(est["SvEstVoltages"])
            emit({"timestamp": ts, "nodes": nodes})  # no pacing on bus

        gapps = GridAPPSD(gappsd_simid)
        assert gapps.connected
        gapps.subscribe(
            service_output_topic("state-estimator", gappsd_simid),
            estimateCallback,
        )
        log.info(
            f"DATA_FEEDER start | GridAPPS-D simid={gappsd_simid} "
            f"| increment={TS_INCREMENT_SEC}s"
        )

        # Callback runs on the listener thread; idle here until COMPLETE.
        while keepLoopingFlag:
            time.sleep(FEEDER_POLL_SEC)

    else:
        # ---------- FILE DRIVER ----------
        # Read the .jsonl straight through, pacing each real record so data
        # doesn't outrun the trainer/forecaster (see FEED_RATE_HZ).
        log.info(
            f"DATA_FEEDER start | path={JSON_PATH} | rate={FEED_RATE_HZ} Hz "
            f"| increment={TS_INCREMENT_SEC}s"
        )
        for record in read_json_records(JSON_PATH):
            emit(record, pace=True)

    # --- end-of-stream: signal completion via the shared Event ---
    # sim_done is the single shutdown signal. Set it after all real records
    # have been enqueued so consumers drain the full tail.
    sim_done.set()
    log.info(
        f"DATA_FEEDER done | total={n_real} real + {n_imp} imputed "
        f"| last_ts={utc_str(last_ts) if last_ts else 'n/a'} | sim_done set"
    )
    train_data_q.cancel_join_thread()
    fc_data_q.cancel_join_thread()
