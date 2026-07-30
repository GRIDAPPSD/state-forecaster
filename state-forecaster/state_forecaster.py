#!/usr/bin/python3

import sys

import torch.multiprocessing as mp

from forecast_feed import feeder_proc
from forecast_train import trainer_proc
from forecast_predict import forecaster_proc


# =====================================================
# MAIN — spawn the three processes, wire the queues.
# =====================================================
def main():
    # GDB 7/20/26: GridAPPS-D simulation ID is first command line argument
    gappsd_simid = None
    if len(sys.argv) > 1:
        gappsd_simid = sys.argv[1]

    mp.set_start_method(
        "spawn", force=True
    )  # required for CUDA + multiprocessing

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
        mp.Process(
            target=feeder_proc,
            args=(train_data_q, fc_data_q, sim_done, gappsd_simid),
            name="feeder",
        ),
        mp.Process(
            target=trainer_proc,
            args=(train_data_q, model_q, sim_done),
            name="trainer",
        ),
        mp.Process(
            target=forecaster_proc,
            args=(fc_data_q, model_q, gappsd_simid),
            name="forecaster",
        ),
    ]

    print(f"[MAIN] spawning {len(procs)} state-forecaster processes")

    for p in procs:
        p.start()

    try:
        # Normal shutdown: feeder finishes → sends DONE → trainer finalizes and
        # sends DONE to model queue → forecaster sees DONE on both → all exit.
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
