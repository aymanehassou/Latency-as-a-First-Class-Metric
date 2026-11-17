"""Transformer encoder -- behaviour recognition on WISDM.

Self-attention over the 128 time steps, with sinusoidal position encoding and a
classification head on the pooled representation. Attention is quadratic in the
window length, which is affordable at W = 128 and would not be at W = 1024.

It is also the most expensive model to TRAIN in the whole benchmark -- around
twenty times the 1D-CNN -- while landing below it on macro-F1. On a dataset of
this size, attention has more capacity than there is signal to use.

    python train_transformer.py
"""
from __future__ import annotations

import config  # noqa: F401
from engine import behaviour

MODEL = "Transformer"

if __name__ == "__main__":
    behaviour.run(MODEL)
