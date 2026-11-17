"""Run ONE latency pass in a fresh process and dump it to a file.

Why this exists: on this Windows/MKL build, calling the measurement pass twice
inside the same interpreter kills the process silently at the start of the
second pass (no traceback -- a native fault, not a Python exception). Running
each pass in its own process avoids the issue entirely and is, if anything,
cleaner: every pass starts from an identical, cold thread pool.

  python bench_pass.py results\lat_pass1.json
"""
from __future__ import annotations

import json
import os
import sys

import bench_latency as B


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python bench_pass.py <out.json>")
    out_path = sys.argv[1]
    res = B.one_pass()
    d = os.path.dirname(os.path.abspath(out_path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2)
    print("saved %s (%d runs)" % (out_path, len(res["runs"])), flush=True)


if __name__ == "__main__":
    main()
