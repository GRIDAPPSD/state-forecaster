#!/usr/bin/python3
"""
check_imputation.py

Directly measure interpolation accuracy of impute_missing_records, independent
of the neural network. Compares each IMPUTED value against the TRUE original
value from the complete file (the value that was dropped to create the gap).

Usage:
    ./check_imputation.py complete.json gappy.json

Where:
    complete.json = the original, gapless file (ground truth)
    gappy.json    = a file produced by drop_timestamps.py from complete.json

Reports MAE (and max abs error) per field (P, Q, V, Angle) over all the
timestamps that were dropped and then reconstructed by the imputer.
"""

import sys
import json

from forecaster_single import (
    read_json_records, impute_missing_records, TS_INCREMENT_SEC,
)


def load_truth(path):
    """ts -> {node -> {P,Q,V,Angle}} for the complete file."""
    truth = {}
    for rec in read_json_records(path):
        truth[int(rec["timestamp"])] = rec["nodes"]
    return truth


def main():
    if len(sys.argv) != 3:
        print("Usage: ./check_imputation.py <complete.json> <gappy.json>")
        sys.exit(1)

    complete_path = sys.argv[1]
    gappy_path = sys.argv[2]

    truth = load_truth(complete_path)

    fields = ["P", "Q", "V", "Angle"]
    sum_abs = {f: 0.0 for f in fields}
    max_abs = {f: 0.0 for f in fields}
    count = 0
    missing_truth = 0

    # Run the imputer over the gappy stream; check only the imputed records.
    source = impute_missing_records(read_json_records(gappy_path), TS_INCREMENT_SEC)
    for rec in source:
        if not rec.get("_imputed", False):
            continue
        ts = int(rec["timestamp"])
        true_nodes = truth.get(ts)
        if true_nodes is None:
            missing_truth += 1
            continue
        for node_name, imp_vals in rec["nodes"].items():
            tv = true_nodes.get(node_name)
            if tv is None:
                continue
            for f in fields:
                err = abs(float(imp_vals[f]) - float(tv[f]))
                sum_abs[f] += err
                if err > max_abs[f]:
                    max_abs[f] = err
            count += 1

    print(f"Imputed node-samples compared: {count}")
    if missing_truth:
        print(f"  (warning: {missing_truth} imputed timestamps had no ground "
              f"truth in complete file)")
    if count == 0:
        print("No imputed records to compare (gappy file may have no gaps).")
        return

    print("\nInterpolation accuracy (imputed vs. true dropped values):")
    print(f"{'field':>6} | {'MAE':>14} | {'max abs err':>14}")
    print("-" * 42)
    for f in fields:
        print(f"{f:>6} | {sum_abs[f]/count:>14.8f} | {max_abs[f]:>14.8f}")


if __name__ == "__main__":
    main()

