"""TCN -- behaviour recognition on WISDM.

A temporal convolutional network: causal, dilated convolutions with residual
connections. The dilation doubles layer by layer, so the receptive field grows
exponentially with depth and covers the whole window with few layers, while
causality means no output ever depends on a future sample -- the property a
streaming detector needs by construction rather than by convention.

The interesting result here is the price: the dilated stack costs several times
a plain 1D-CNN per window without buying accuracy on this task. The receptive
field was already large enough.

    python train_tcn.py
"""
from __future__ import annotations

import config  # noqa: F401
from engine import behaviour

MODEL = "TCN"

if __name__ == "__main__":
    behaviour.run(MODEL)
