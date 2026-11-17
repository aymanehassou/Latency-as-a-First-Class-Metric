"""Merge several single-pass latency files into results/latency.json.

Same aggregation as bench_latency.main(): median across passes for every timing
column, plus the min/max and the inter-pass spread, which is the honest way to
report an absolute millisecond figure from a machine you do not control.

  python merge_latency.py results\lat_pass*.json
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

from bench_latency import AGG_KEYS, OUT


def main():
    pats = sys.argv[1:] or [os.path.join("results", "lat_pass*.json")]
    files = []
    for p in pats:
        files += sorted(glob.glob(p))
    files = sorted(set(files))
    if not files:
        sys.exit("no pass files matched %s" % pats)

    passes = []
    for fp in files:
        with open(fp) as fh:
            passes.append(json.load(fh))
    print("merging %d passes:" % len(passes))
    for fp in files:
        print("   %s" % fp)

    base = passes[0]
    merged = []
    for r in base["runs"]:
        key = (r["config"], r["model"])
        vals = {k: [] for k in AGG_KEYS}
        for p in passes:
            rr = next((x for x in p["runs"] if (x["config"], x["model"]) == key), None)
            if rr is None:
                continue
            for k in AGG_KEYS:
                if k in rr:
                    vals[k].append(float(rr[k]))
        rec = dict(r)
        for k, v in vals.items():
            if v:
                rec[k] = float(np.median(v))
        p50s = vals["lat_p50_ms"]
        rec["lat_p50_ms_min"] = float(min(p50s))
        rec["lat_p50_ms_max"] = float(max(p50s))
        rec["lat_p50_ms_spread_pct"] = float(
            100.0 * (max(p50s) - min(p50s)) / np.median(p50s))
        merged.append(rec)

    out = {"machine": base["machine"], "repeats": len(passes), "runs": merged}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    worst = max(merged, key=lambda r: r["lat_p50_ms_spread_pct"])
    print("saved %s  (median of %d passes; worst inter-pass spread %.1f%% on %s/%s)"
          % (OUT, len(passes), worst["lat_p50_ms_spread_pct"], worst["config"],
             worst["model"]))
    for r in merged:
        print("  %-6s %-16s p50=%8.3f ms  (min %.3f / max %.3f)"
              % (r["config"], r["model"], r["lat_p50_ms"],
                 r["lat_p50_ms_min"], r["lat_p50_ms_max"]))


if __name__ == "__main__":
    main()
