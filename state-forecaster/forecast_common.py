"""
forecast_common.py — Shared, dependency-light utilities for the State Forecaster.

Torch-free by design: this module is imported by every process (data_feeder,
trainer, forecaster), so it must not pull in heavy ordomain-specific
dependencies. It holds only cross-cutting configuration constants and small
helpers used across process boundaries.
"""

import logging

from datetime import datetime, timezone

# Directory for per-process log files (see setup_logger).
LOG_DIR = "./logs"

# --- streaming cadence ---
# Spacing, in seconds, between consecutive state-estimate timestamps in the
# input stream. MUST match the actual cadence of the data source (the cadence
# guard in the data_feeder enforces this at runtime). Also must divide evenly
# into the day/week lag intervals. Uncomment the value matching the run:
# TS_INCREMENT_SEC = 3     # real-time simulation cadence
# TS_INCREMENT_SEC = 60    # 1-minute increment
TS_INCREMENT_SEC = 300  # 5-minute increment
# TS_INCREMENT_SEC = 900   # 15-minute increment

# Sentinel value placed on inter-process queues to signal end-of-stream.
# (A distinguished queue item, not an OS signal — consumers stop on receipt.)
DONE = "__DONE__"


def utc_str(epoch_sec):
    """Format an epoch-seconds timestamp as a UTC 'YYYY-MM-DD HH:MM:SS' string.
    Used for human-readable timestamps in log output across all processes."""
    return datetime.fromtimestamp(epoch_sec, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# =====================================================
# LOGGING (per-process: each process configures its own file + console)
# =====================================================
def setup_logger(name, logfile):
    """Create a logger that writes to both a per-process file and the console.

    Each process (data_feeder, trainer, forecaster) calls this with its own
    name and logfile so their output can be followed independently (e.g.
    `tail -f`). Clears existing handlers so re-creation in a spawned child
    process doesn't duplicate log lines.

    Args:
        name:    logger name (also used as the process identifier).
        logfile: path to this process's log file (overwritten each run, mode="w").

    Returns:
        A configured logging.Logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s [%(processName)s] %(message)s", datefmt="%H:%M:%S"
    )
    fh = logging.FileHandler(logfile, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger
