import logging

from datetime import datetime, timezone

LOG_DIR = "./logs"

# --- streaming cadence ---
# TS_INCREMENT_SEC = 3        # timestamp spacing in the input stream (real-time)
# TS_INCREMENT_SEC = 60       # timestamp spacing in the input stream (1 min)
TS_INCREMENT_SEC = 300  # timestamp spacing in the input stream (5 min)
# TS_INCREMENT_SEC = 900      # timestamp spacing in the input stream (15 min)

DONE = "__DONE__"  # sentinel Queue item meaning "end of stream"


def utc_str(epoch_sec):
    return datetime.fromtimestamp(epoch_sec, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# =====================================================
# LOGGING (per-process: each process configures its own file + console)
# =====================================================
def setup_logger(name, logfile):
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
