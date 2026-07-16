#!/usr/bin/python3
"""
simple_to_real_json.py

Convert the "simplified" state-estimate JSON Lines format into the "real"
GridAPPS-D State Estimator publish format (SvEstVoltages), one JSON object
per line (one per timestamp).

FIRST-CUT SIMPLIFICATIONS (deliberate; see project notes):
  * ConnectivityNode keeps the FULL simplified node name INCLUDING the phase
    suffix (e.g. "632.1"), so it remains unique per entry on its own. The
    forecaster (for now) uses this combined name as its single node key.
    (In the true bus format ConnectivityNode is a phase-less CIM mRID shared
    across up to 3 phase entries; we intentionally differ here.)
  * A separate "phase" field is ALSO added (1->A, 2->B, 3->C), so phase info
    is duplicated (inside ConnectivityNode and in phase). This carries the
    phase forward for eventual GridAPPS-D consistency.
  * NO unit conversion on angle. Values are passed through UNCHANGED (treated
    as radians end-to-end). The real bus format publishes degrees, but per
    project decision we keep radians here; this also makes the transform
    lossless (a later new-format reader can reproduce byte-identical records).
  * Variance fields (angleVariance, vVariance) are omitted (unused).

Field mapping (per node entry):
    simplified key "632.1"  -> ConnectivityNode "632.1"  (kept whole)
                     suffix  -> phase  (1->A, 2->B, 3->C)
    P     -> P     (pass through)
    Q     -> Q     (pass through)
    V     -> v     (pass through; note lowercase key)
    Angle -> angle (pass through, NO unit change; note lowercase key)

Output record shape (one per line):
    {"SvEstVoltages": [
        {"ConnectivityNode": "632.1", "phase": "A",
         "P": ..., "Q": ..., "v": ..., "angle": ...},
        ...
     ],
     "timeStamp": 1700000000}

Usage:
    ./simple_to_real_json.py input_simplified.json output_real.json
"""

import sys
import json

# phase number (as string) -> phase letter
PHASE_NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C"}
# defensive: also accept letter suffixes already in a/b/c form
PHASE_LETTER_TO_LETTER = {"a": "A", "b": "B", "c": "C"}


def derive_phase(node_key):
    """Return the phase letter (A/B/C) parsed from a simplified node key's
    suffix after the LAST dot. Returns "" (with a warning) if no dot or an
    unrecognized suffix."""
    if "." not in node_key:
        print(f"[WARN] node key '{node_key}' has no '.' suffix; "
              f"emitting empty phase.")
        return ""
    suffix = node_key.rsplit(".", 1)[1]
    if suffix in PHASE_NUM_TO_LETTER:
        return PHASE_NUM_TO_LETTER[suffix]
    if suffix.lower() in PHASE_LETTER_TO_LETTER:
        return PHASE_LETTER_TO_LETTER[suffix.lower()]
    print(f"[WARN] node key '{node_key}' has unrecognized phase suffix "
          f"'{suffix}'; emitting empty phase.")
    return ""


def convert(in_path, out_path):
    n_lines = 0
    n_entries = 0
    with open(in_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            if not line:
                continue
            rec = json.loads(line)
            ts = int(rec["timestamp"])
            nodes = rec["nodes"]

            sv_entries = []
            for node_key, vals in nodes.items():
                phase = derive_phase(node_key)
                sv_entries.append({
                    "ConnectivityNode": node_key,   # kept whole (incl. phase)
                    "phase": phase,
                    "P": vals["P"],
                    "Q": vals["Q"],
                    "v": vals["V"],                 # V -> v
                    "angle": vals["Angle"],         # Angle -> angle (radians, unchanged)
                })
                n_entries += 1

            out_rec = {"SvEstVoltages": sv_entries, "timeStamp": ts}
            fout.write(json.dumps(out_rec) + "\n")
            n_lines += 1

    print(f"Read {n_lines} timestamp records ({n_entries} node entries).")
    print(f"Wrote real-format JSON Lines to: {out_path}")


def main():
    if len(sys.argv) != 3:
        print("Usage: ./simple_to_real_json.py <input_simplified.json> "
              "<output_real.json>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()

