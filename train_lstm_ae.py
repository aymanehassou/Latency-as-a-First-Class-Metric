"""LSTM auto-encoder -- state detection on SKAB and SMD.

An LSTM encodes the window into a final hidden state; a second LSTM unrolls
that state back into a sequence. The bottleneck is temporal rather than
spatial, so what it fails to reconstruct is a dynamic the nominal data never
contained -- a drift, an oscillation, a transient -- rather than an unusual
instantaneous value.

In exchange, inference is sequential and roughly twice the convolutional
auto-encoder's cost per window. On SMD, where a sample arrives once a minute,
that is irrelevant; on a fast sensor loop it would not be.

    python train_lstm_ae.py
"""
from __future__ import annotations

import config  # noqa: F401
from engine import state

MODEL = "LSTM-AE"

if __name__ == "__main__":
    state.run(MODEL)
