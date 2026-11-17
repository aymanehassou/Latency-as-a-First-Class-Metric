"""Transformer auto-encoder -- state detection on SKAB and SMD.

A self-attention encoder compresses the window; a light decoder reconstructs
it. Attention can relate two distant samples directly instead of passing
information along a chain of hidden states, which in principle suits long
windows and multi-channel coupling.

In practice it ends up tied with the LSTM auto-encoder on both datasets, which
is itself the finding: on SKAB and SMD the ceiling is set by the thresholding
problem, not by the model's ability to represent the nominal regime.

    python train_transformer_ae.py
"""
from __future__ import annotations

import config  # noqa: F401
from engine import state

MODEL = "Transformer-AE"

if __name__ == "__main__":
    state.run(MODEL)
