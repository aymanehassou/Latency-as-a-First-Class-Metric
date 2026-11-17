"""Project-wide paths and defaults.

Import this FIRST in every entry point, before anything from `rtbench`:
the data loader reads its root from the environment at import time, so the
variable has to be in place before that module is loaded.

    import config          # noqa: F401  -- sets RTBENCH_DATA
    from engine import behaviour

Nothing here is read from the command line, on purpose: where the data comes
from and how each track is configured can be seen by opening one file.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.abspath(__file__))

#: The three public datasets (~530 MB) are not part of the repository.
#: `python fetch_data.py` downloads them into ./data; set the environment
#: variable RTBENCH_DATA to use a copy stored elsewhere.
#: Layout expected underneath:
#:    wisdm/WISDM_ar_v1.1_raw.txt
#:    skab/{anomaly-free,valve1,valve2,other}/*.csv
#:    smd/{train,test,test_label}/machine-*.txt
DATA_DIR = os.environ.get("RTBENCH_DATA", os.path.join(ROOT, "data"))

#: One JSON per model lands here; `collect.py` merges them into the three
#: aggregate files the analysis expects.
RESULTS_DIR = os.path.join(ROOT, "results")

#: LaTeX tables, figures and macros regenerated from THESE results.
PAPER_OUT = os.path.join(ROOT, "paper_out")

os.environ.setdefault("RTBENCH_DATA", DATA_DIR)
os.environ.setdefault("RTBENCH_RESULTS", RESULTS_DIR)
os.environ.setdefault("RTBENCH_PAPER", PAPER_OUT)

os.makedirs(RESULTS_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# Experimental protocol -- identical to the manuscript
# --------------------------------------------------------------------------- #

#: Behaviour track (WISDM, 20 Hz accelerometer, 6 activities, 36 subjects).
#: 128 samples = 6.4 s; the decision is attributed to the LAST sample of the
#: window, never its centre -- centring would grant W/2 samples of lookahead
#: and make every reported delay optimistic.
WISDM = dict(win=128, stride=64, epochs=30, batch_size=128, lr=2e-3,
             seeds=(0, 1, 2))

#: State track, dataset 1 (SKAB, 1 Hz, 8 channels, 34 runs of a pump rig).
#: 100 nominal samples at the head of each run are reserved as a label-free
#: warm-up for per-run threshold calibration and are excluded from scoring
#: under every policy, so the policies are compared on identical samples.
SKAB = dict(win=30, stride_train=1, calib_len=100, epochs=30, batch_size=128,
            lr=1e-3, seeds=(0, 1, 2), val_quantile=99.0)

#: State track, dataset 2 (SMD, 1 sample/min, 38 channels, 28 server entities).
#: One model is trained per entity, so a single seed already means 28 fits per
#: architecture. SMD ships its own nominal split, which is why the `global` and
#: `runcal` policies coincide there by construction.
SMD = dict(win=100, stride_train=10, epochs=20, batch_size=128, lr=1e-3,
           seeds=(0,), val_quantile=99.0)

#: Training device. Inference latency is ALWAYS measured single-thread on the
#: CPU, whatever this says -- see rev_latency_real.py.
DEVICE = "cuda"

#: Torch threads during training (the GPU does the work; these feed it).
THREADS = 4


def banner(model: str, track: str) -> None:
    print("=" * 72)
    print("  model   : %s" % model)
    print("  track   : %s" % track)
    print("  data    : %s" % DATA_DIR)
    print("  results : %s" % RESULTS_DIR)
    print("  device  : %s" % DEVICE)
    print("=" * 72, flush=True)
