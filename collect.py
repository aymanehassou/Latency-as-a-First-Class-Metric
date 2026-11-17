"""Merge the per-model result files into the three aggregates the analysis reads.

Each training script writes its own file, so that any single model can be
re-run without disturbing the others:

    results/wisdm_<model>.json     {dataset: {...}, runs: [...]}
    results/skab_<model>.json      [record, ...]
    results/smd_<model>.json       [record, ...]

This script concatenates them, in the order the manuscript lists the models,
into the files `tools/analyze.py` expects:

    results/wisdm.json   results/skab.json   results/smd_A.json

It also reports what is missing, so an interrupted run is obvious rather than
silently producing a table with a hole in it.

    python collect.py
"""
from __future__ import annotations

import json
import os
import sys

import config

RES = config.RESULTS_DIR

BEHAVIOUR = ["RF-features", "MLP", "CNN1D", "TCN", "LSTM", "BiGRU", "Transformer"]
STATE = ["PCA", "IsolationForest", "DenseAE", "ConvAE", "LSTM-AE", "Transformer-AE"]


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def collect_behaviour():
    merged, dataset, missing = [], None, []
    for m in BEHAVIOUR:
        d = _load(os.path.join(RES, "wisdm_%s.json" % m))
        if d is None:
            missing.append(m)
            continue
        dataset = dataset or d["dataset"]
        merged += d["runs"]
    if dataset is None:
        print("WISDM: nothing to collect")
        return missing
    out = os.path.join(RES, "wisdm.json")
    with open(out, "w") as fh:
        json.dump({"dataset": dataset, "runs": merged}, fh, indent=2)
    print("WISDM : %2d runs from %d models -> %s"
          % (len(merged), len(BEHAVIOUR) - len(missing), out))
    return missing


def collect_state(tag, out_name):
    merged, missing = [], []
    for m in STATE:
        d = _load(os.path.join(RES, "%s_%s.json" % (tag, m)))
        if d is None:
            missing.append(m)
            continue
        merged += d
    if not merged:
        print("%s: nothing to collect" % tag.upper())
        return missing
    out = os.path.join(RES, out_name)
    with open(out, "w") as fh:
        json.dump(merged, fh, indent=2)
    print("%-6s: %3d records from %d models -> %s"
          % (tag.upper(), len(merged), len(STATE) - len(missing), out))
    return missing


def main():
    miss_w = collect_behaviour()
    miss_sk = collect_state("skab", "skab.json")
    miss_sm = collect_state("smd", "smd_A.json")

    gaps = []
    for label, miss in (("WISDM", miss_w), ("SKAB", miss_sk), ("SMD", miss_sm)):
        if miss:
            gaps.append("%s: %s" % (label, ", ".join(miss)))
    if gaps:
        print("\nSTILL MISSING")
        for g in gaps:
            print("   " + g)
        print("\nRun the corresponding train_*.py, then collect.py again.")
        sys.exit(1)
    print("\nAll models present. Next: tools\\run_analysis.bat")


if __name__ == "__main__":
    main()
