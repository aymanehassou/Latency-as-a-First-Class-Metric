"""Make the full-pipeline latency measurement the one the paper reports.

bench_latency.py times normalisation + forward pass on one synthetic window for
the neural models. rev_latency_real.py times the same, complete decision for
every detector on distinct real test windows; that is the measurement the
paper reports. This script swaps it in:

    results/latency.json               <- results/latency_real.json
    results/latency_forward_only.json  <- the forward-pass-only measurement

It is idempotent and never loses the original measurement.
"""
from __future__ import annotations

import json
import os
import shutil

import rev_common as R

SRC = os.path.join(R.RESULTS, "latency_real.json")
DST = os.path.join(R.RESULTS, "latency.json")
BAK = os.path.join(R.RESULTS, "latency_forward_only.json")


def main():
    if not os.path.exists(SRC):
        raise SystemExit("run rev_summarize.py first: %s is missing" % SRC)
    cur = json.load(open(DST))
    if "protocol" not in cur and not os.path.exists(BAK):
        shutil.copyfile(DST, BAK)
        print("kept the forward-pass-only measurement as", BAK)
    new = json.load(open(SRC))
    with open(DST, "w") as fh:
        json.dump(new, fh, indent=2)
    print("latency.json now holds: %s (%d detectors, median of %d passes)"
          % (new.get("protocol"), len(new["runs"]), new.get("repeats", 0)))


if __name__ == "__main__":
    main()
