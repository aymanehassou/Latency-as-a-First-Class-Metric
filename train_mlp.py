"""MLP -- behaviour recognition on WISDM.

The window is flattened and fed to two fully connected layers. It therefore has
no notion of time at all: swapping two samples inside the window would not
change its answer. That is exactly why it belongs in the benchmark -- it is the
floor that any architecture claiming to "model temporal structure" has to beat,
and it is by far the cheapest model here, which makes it the reference point on
the accuracy/latency plane.

    python train_mlp.py
"""
from __future__ import annotations

import config  # noqa: F401  -- must be imported before anything from rtbench
from engine import behaviour

MODEL = "MLP"

if __name__ == "__main__":
    behaviour.run(MODEL)
