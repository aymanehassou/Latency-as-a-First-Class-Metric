"""Isolation Forest -- the non-reconstruction reference for state detection.

A hundred random isolation trees over five summary statistics per channel. It
does not reconstruct anything: it scores how few random splits are needed to
isolate a point, which is a different notion of "unusual" from reconstruction
error and is worth having in the comparison for that reason.

It produces the lowest false-alarm rate of any detector here and the lowest
point-wise F1, because it alarms rarely and late -- a median delay of about
half a minute on SKAB. Read together, those two numbers say what a single F1
cannot: this detector is conservative, not accurate.

It is also, at roughly 28 ms per window, the slowest model in the benchmark on
this machine -- tree traversal is branch-prediction bound and does not vectorise.

    python train_iforest.py
"""
from __future__ import annotations

import config  # noqa: F401
from engine import state

MODEL = "IsolationForest"

if __name__ == "__main__":
    state.run(MODEL)
