#!/usr/bin/python3
"""
state_forecaster.py — Entry point / launcher for the State Forecaster app.

Spawns and wires together the three worker processes that make up the app:

    data_feeder : reads state estimates (GridAPPS-D bus or file), imputes gaps,
                  and distributes each record to the trainer and forecaster.
    trainer     : accumulates 2-day blocks, trains the DNN, publishes model
                  snapshots to the forecaster.
    forecaster  : ingests estimates, forecasts the latest timestamp using the
                  newest model snapshot, publishes/records forecasts.

The processes communicate only through the queues and the sim_done Event
created here; they share no other state. Invoked with an optional GridAPPS-D
simulation ID (bus mode); with no argument it runs in file mode.
"""

import sys

import torch.multiprocessing as mp

from forecast_feed import feeder_proc
from forecast_train import trainer_proc
from forecast_predict import forecaster_proc


# =====================================================
# MAIN — spawn the three processes, wire the queues.
# =====================================================
def main():
    """Create the shared queues/Event, spawn the three worker processes, and
    manage their lifecycle (normal join, Ctrl-C teardown, exit-code report)."""
    # GDB 7/20/26: GridAPPS-D simulation ID is first command line argument.
    # Present -> bus mode (subscribe to that simulation); absent -> file mode.
    gappsd_simid = None
    if len(sys.argv) > 1:
        gappsd_simid = sys.argv[1]

    # "spawn" (not fork) is required for CUDA + multiprocessing: each child
    # initializes its own CUDA context cleanly.
    mp.set_start_method("spawn", force=True)

    # --- Queues + end-of-simulation Event ---
    # train_data_q : feeder -> trainer.   Keep-ALL FIFO so the trainer's
    #                RollingBuffer sees a gapless history (no dropped records).
    # fc_data_q    : feeder -> forecaster. Keep-ALL FIFO too (gapless history);
    #                the forecaster forec but must ingest
    #                every record to keep its recent-history window contiguous.
    # model_q      : trainer PUTs model snapshots; forecaster GETs them.
    #                Consumer takes latest-only via drain_latest (snapshots are
    #                infrequent, so an unbounded queue stays shallow).
    # sim_done     : one-shot Event the feeder sets at end-of-stream so the
    #                trainer and forecaster can exit cleanly.
    train_data_q = mp.Queue()
    fc_data_q = mp.Queue()
    model_q = mp.Queue()
    sim_done = mp.Event()

    procs = [
        mp.Process(
            target=feeder_proc,
            args=(train_data_q, fc_data_q, sim_done, gappsd_simid),
            name="data_feeder",
        ),
        mp.Process(
            target=trainer_proc,
            args=(train_data_q, model_q, sim_done),
            name="trainer",
        ),
        mp.Process(
            target=forecaster_proc,
            args=(fc_data_q, model_q, sim_done, gappsd_simid),
            name="forecaster",
        ),
    ]

    print(f"[MAIN] spawning {len(procs)} state-forecaster processes")

    for p in procs:
        p.start()

    try:
        for p in procs:
            p.join()
        bad = [p for p in procs if p.exitcode not in (0, None)]
        if bad:
            print(
                f"[MAIN] processes exited with errors: "
                f"{[(p.name, p.exitcode) for p in bad]}"
            )
        else:
            print("[MAIN] all processes exited cleanly.")

    except KeyboardInterrupt:
        # Ctrl-C: terminate children so we don't leave orphaned processes.
        print("\n[MAIN] KeyboardInterrupt → terminating child processes...")
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
        print("[MAIN] shutdown complete.")

    finally:
        # Report any non-zero exit codes (a crashed child surfaces here).
        for p in procs:
            if p.exitcode not in (0, None):
                print(f"[MAIN] WARNING: {p.name} exited with code {p.exitcode}")


if __name__ == "__main__":
    main()
