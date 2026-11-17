"""1D-CNN -- behaviour recognition on WISDM.

Stacked 1D convolutions over the time axis followed by global pooling. Local
shape is what distinguishes walking from climbing stairs, and convolution is
the cheapest way to look for local shape: every output depends on a short
neighbourhood, and the whole window is processed in one dense pass that maps
well onto a CPU.

This is the model the paper recommends. Not because it has the best score --
BiGRU is within noise of it -- but because it reaches that score at a fraction
of the per-window cost, which is the only comparison that means anything for a
system that has to answer in real time.

    python train_cnn1d.py
"""
from __future__ import annotations

import config  # noqa: F401
from engine import behaviour

MODEL = "CNN1D"

if __name__ == "__main__":
    behaviour.run(MODEL)
