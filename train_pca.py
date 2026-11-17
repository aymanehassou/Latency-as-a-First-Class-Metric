"""PCA reconstruction -- the linear reference for state detection.

Sixteen principal components fitted on nominal windows; the score is the
squared error of the reconstruction from that subspace. There is no training
loop and no gradient: it fits in under a second and runs in about seventy
microseconds per window, which makes it the cheapest detector in the benchmark
by two orders of magnitude.

It is here to keep the deep reconstructors honest. A neural auto-encoder that
does not clearly beat a 16-component linear subspace has not earned its GPU.

    python train_pca.py
"""
from __future__ import annotations

import config  # noqa: F401
from engine import state

MODEL = "PCA"

if __name__ == "__main__":
    state.run(MODEL)
