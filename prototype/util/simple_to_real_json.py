#!/usr/bin/python3
"""
simple_to_real_json.py

Convert the "simplified" state-estimate JSON Lines format into the "real"
GridAPPS-D State Estimator publish format (SvEstVoltages), one JSON object
per line (one per timestamp).

NODE / PHASE HANDLING:
  * The simplified node key (e.g. "632.1") is split on its LAST dot into a
    phase-less ConnectivityNode ("632") and a separate phase ("1").
  * The phase is passed through AS-IS — NO number<->letter mapping. Whatever
    the input uses ("1", "A", etc.) is emitted unchanged, so the phase we
    publish is guaranteed identical to the phase we received. (Dots are only
    ever separators, never part of a ConnectivityNode value.)

OTHER NOTES:
  * NO unit conversion on angle. Values pass through UNCHANGED (radians
    end-to-end). This keeps the transform lossless: the forecaster's reader
    recombines ConnectivityNode + "." + phase into the original internal key,
    so forecasts remain byte-identical to the old simplified-file runs.
  * Variance fields (angleVariance, vVariance) are omitted (unused).

Field mapping (per node entry):
    simplified key "632.1"  -> ConnectivityNode "632"  +  phase "1"
    P     -> P     (pass through)
    Q     -> Q     (pass through)
    V     -> v     (pass through; note lowercase key)
    Angle -> angle (pass through, NO unit change; note lowercase key)

Output record shape (one per line):
    {"SvEstVoltages": [
        {"ConnectivityNode": "632", "phase": "1",
         "P": ..., "Q": ..., "v": ..., "angle": ...},
        ...
     ],
     "timeStamp": 1700000000}

Usage:
    ./simple_to_real_json.py input_simplified.json output_real.json
"""

import sys
import json


def split_node_key(node_key):
    """Split a simplified node key into (ConnectivityNode, phase) on the LAST
    dot. Phase is returned AS-IS (no mapping). If there is no dot, returns the
    whole key as ConnectivityNode and "" as phase, with a warning."""
    if "." not in node_key:
        print(f"[WARN] node key '{node_key}' has no '.' separator; "
              f"emitting phase-less entry.")
        return node_key, ""
    cn, phase = node_key.rsplit(".", 1)
    return cn, phase


def convert(in_path, out_path):
    n_lines = 0
    n_entries = 0
    with open(in_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ts = int(rec["timestamp"])
            nodes = rec["nodes"]

            sv_entries = []
            for node_key, vals in nodes.items():
                cn, phase = split_node_key(node_key)
                sv_entries.append({
                    "ConnectivityNode": cn,       # phase-less
                    "phase": phase,               # as-is, no mapping
                    "P": vals["P"],
                    "Q": vals["Q"],
                    "v": vals["V"],               # V -> v
                    "angle": vals["Angle"],       # Angle -> angle (radians, unchanged)
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

