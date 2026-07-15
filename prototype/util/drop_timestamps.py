#!/usr/bin/python3
"""
drop_timestamps.py

Create a "gappy" copy of a line-delimited JSON state-estimate file by randomly
dropping a fraction of records, to test the forecaster's imputation handling.

Rules that mirror the real system's assumptions:
  * The FIRST record is always kept (it is "time zero"; no leading imputation).
  * Dropped timestamps remain grid-aligned by construction (we only ever remove
    whole records; the survivors keep their original grid-aligned timestamps).
  * Reports the largest resulting consecutive gap so you can gauge how hard the
    imputer will have to work.

Usage:
    ./drop_timestamps.py input.json output.json 0.20 [seed]
        -> drops ~20% of records (except the first), writing the rest in order.
"""

import sys
import json
import random


def main():
    if len(sys.argv) < 4:
        print("Usage: ./drop_timestamps.py <input.json> <output.json> "
              "<drop_fraction 0..1> [seed]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    drop_frac = float(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    random.seed(seed)

    kept = 0
    dropped = 0
    first = True
    prev_ts = None
    max_gap_steps = 1
    increment = None

    with open(in_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ts = int(record["timestamp"])

            # infer the grid increment from the first two records
            if prev_ts is not None and increment is None:
                increment = ts - prev_ts

            keep = first or (random.random() >= drop_frac)
            if keep:
                fout.write(json.dumps(record) + "\n")
                kept += 1
                # track largest consecutive gap among KEPT records
                if increment and prev_ts is not None:
                    gap_steps = (ts - prev_ts) // increment
                    if gap_steps > max_gap_steps:
                        max_gap_steps = gap_steps
                prev_ts = ts   # prev_ts tracks last KEPT timestamp
            else:
                dropped += 1

            first = False

    total = kept + dropped
    print(f"Read {total} records | kept {kept} | dropped {dropped} "
          f"({100.0*dropped/total:.1f}%)")
    print(f"Inferred increment: {increment} s")
    print(f"Largest consecutive gap among kept records: {max_gap_steps} steps "
          f"({max_gap_steps * (increment or 0)} s)")
    print(f"Wrote gappy file: {out_path}")


if __name__ == "__main__":
    main()

