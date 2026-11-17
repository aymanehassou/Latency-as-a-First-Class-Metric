"""Convolutional auto-encoder -- state detection on SKAB and SMD.

Strided 1D convolutions compress the window, transposed convolutions rebuild
it. Weight sharing across time means it reconstructs local waveform shape with
far fewer parameters than the dense auto-encoder, and it is the cheapest neural
reconstructor per window on SMD's 38 channels.

    python train_conv_ae.py
"""
from __future__ import annotations

import config  # noqa: F401
from engine import state

MODEL = "ConvAE"

if __name__ == "__main__":
    state.run(MODEL)
