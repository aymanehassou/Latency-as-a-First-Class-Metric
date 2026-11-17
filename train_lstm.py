"""LSTM -- behaviour recognition on WISDM.

A recurrent network that walks the window one sample at a time, carrying a
hidden state. Unlike the convolutional models it is inherently sequential: the
128 steps cannot be computed in parallel, which is why its per-window latency
is an order of magnitude above the 1D-CNN's for a comparable parameter count.
Parameter count and inference cost are not the same quantity, and the recurrent
models are where that gap is widest.

    python train_lstm.py
"""
from __future__ import annotations

import config  # noqa: F401
from engine import behaviour

MODEL = "LSTM"

if __name__ == "__main__":
    behaviour.run(MODEL)
