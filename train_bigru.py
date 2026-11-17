"""BiGRU -- behaviour recognition on WISDM.

A bidirectional GRU: one pass forward through the window, one pass backward,
the two hidden states concatenated. It is usually the strongest model on this
dataset, and it is here to make the paper's central point concrete.

Two things to keep in mind when reading its score.

First, bidirectionality is legitimate here only because the decision is taken
at the END of a completed window -- the backward pass reads samples that have
already arrived, not future ones. Applied to a centred window it would be
lookahead, and the delay figures would be fiction.

Second, it is by a wide margin the most expensive model in the benchmark:
roughly forty times the 1D-CNN's per-window cost for an accuracy difference
that sits inside the seed-to-seed spread. On a 20 Hz stream it still fits, but
the real-time margin is the thinnest of any model here.

    python train_bigru.py
"""
from __future__ import annotations

import config  # noqa: F401
from engine import behaviour

MODEL = "BiGRU"

if __name__ == "__main__":
    behaviour.run(MODEL)
