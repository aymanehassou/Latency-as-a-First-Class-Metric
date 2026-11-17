"""Random Forest on hand-crafted features -- the pre-deep-learning reference.

Eighteen statistics per channel (mean, standard deviation, min, max, quartiles,
median absolute difference, RMS, dominant FFT bin) fed to a 200-tree forest.
This is what activity recognition looked like before end-to-end learning, and
no benchmark that omits it can claim a deep model was worth the trouble.

It trains in seconds and reaches a macro-F1 the MLP does not beat. What it does
not do is run fast at inference: feature extraction plus 200 tree traversals is
the slowest single-window path in the entire benchmark -- slower than every
neural model including the BiGRU. That inversion is one of the paper's results,
and it is only visible because latency is measured rather than assumed.

    python train_rf.py
"""
from __future__ import annotations

import config  # noqa: F401
from engine import behaviour

MODEL = "RF-features"

if __name__ == "__main__":
    behaviour.run(MODEL)
