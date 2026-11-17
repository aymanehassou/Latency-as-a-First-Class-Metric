"""Dense auto-encoder -- state detection on SKAB and SMD.

The window is flattened, squeezed through a narrow bottleneck and reconstructed.
Anything the bottleneck cannot represent -- because nothing like it appeared in
the nominal training data -- comes back with a large reconstruction error, and
that error is the anomaly score.

Like the MLP on the behaviour track, it ignores time entirely, which makes it
the control: if a recurrent or convolutional auto-encoder cannot beat it, the
extra machinery was not paying for itself. On these two datasets the spread
between all four neural reconstructors turns out to be small -- the choice of
threshold matters far more than the choice of architecture.

    python train_dense_ae.py
"""
from __future__ import annotations

import config  # noqa: F401
from engine import state

MODEL = "DenseAE"

if __name__ == "__main__":
    state.run(MODEL)
