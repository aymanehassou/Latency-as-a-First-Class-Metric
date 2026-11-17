"""Training variance on SMD.

The published SMD numbers use one seed per entity. This script retrains every
detector with three seeds on a fixed subset of six entities (two per machine
group) and stores the deployable and oracle results per seed, so the spread
across seeds can be reported.

Output: results/rev_seeds_smd.json   (resumable: finished items are skipped)
"""
from __future__ import annotations

import numpy as np

import rev_common as R
from rev_common import D, L

ENTITIES = ["machine-1-1", "machine-1-5", "machine-2-1", "machine-2-5",
            "machine-3-1", "machine-3-6"]
SEEDS = (0, 1, 2)
Q = float(R.cfg("SMD", "val_quantile", 99.0))
OUT = "rev_seeds_smd.json"


def main():
    L.set_threads(R.threads())
    dev = R.device()
    win = int(R.cfg("SMD", "win", 100))
    epochs = int(R.cfg("SMD", "epochs", 20))
    stride = int(R.cfg("SMD", "stride_train", 10))
    bs, lr = int(R.cfg("SMD", "batch_size", 128)), float(R.cfg("SMD", "lr", 1e-3))

    have = set(D.smd_entities())
    ents = [e for e in ENTITIES if e in have]
    if len(ents) != len(ENTITIES):
        print("[seeds] WARNING: missing entities", sorted(set(ENTITIES) - have))
    prev = R.load(OUT)
    rows = prev.get("rows", []) if prev else []
    done = {(r["entity"], r["model"], r["seed"]) for r in rows}
    print(f"[seeds] SMD entities {ents} seeds {SEEDS} device {dev} epochs {epochs} "
          f"stride_train {stride}", flush=True)

    def save():
        return R.dump({"entities": ents, "seeds": SEEDS, "q": Q, "win": win, "epochs": epochs,
                       "stride_train": stride, "device": dev, "rows": rows}, OUT)

    for ent in ents:
        todo = [(n, s) for s in SEEDS for n in R.STATE_MODELS if (ent, n, s) not in done]
        if not todo:
            continue
        print(f"{R.stamp()} === {ent}: loading", flush=True)
        ds = D.load_smd_entity(ent, win=win, stride_train=stride)
        hz = ds["meta"]["sampling_hz"]
        seqs, trX, vaX = ds["test_seqs"], ds["train"].X, ds["val"].X
        for name, seed in todo:
            val_sc, seq_sc, tr = R.fit_and_score(name, seed, trX, vaX, seqs, epochs, bs, lr, dev)
            thr_val = float(np.percentile(val_sc, Q))
            dep = R.slim(R.evaluate_scores(seq_sc, seqs, thr_val, hz))
            orc = R.slim(R.evaluate_scores(seq_sc, seqs, R.oracle_threshold(seq_sc, seqs), hz))
            rows.append({"entity": ent, "model": name, "seed": seed, "deployable": dep,
                         "oracle": orc, "train_seconds": tr["train_seconds"]})
            print(f"{R.stamp()}   {name:15s} seed {seed}: F1 dep {dep['f1']:.3f} "
                  f"orc {orc['f1']:.3f} ER dep {dep['er']:.2f}", flush=True)
            save()
    print("saved", save())


if __name__ == "__main__":
    main()
