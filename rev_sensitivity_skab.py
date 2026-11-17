"""Sensitivity of the per-run policy to q and N_c on SKAB.

Each detector is trained once per seed with the published recipe; its scores
are then thresholded for every q in QS and every warm-up length in NCS.
The scored samples are the same for every setting (the first 100 samples of
each run are always excluded), so only the threshold changes.

Output: results/rev_sens_skab.json
"""
from __future__ import annotations

import numpy as np

import rev_common as R
from rev_common import D, L

QS = (97.5, 99.0, 99.5)
NCS = (50, 75, 100)
OUT = "rev_sens_skab.json"


def main():
    L.set_threads(R.threads())
    dev = R.device()
    win = int(R.cfg("SKAB", "win", 30))
    epochs = int(R.cfg("SKAB", "epochs", 30))
    seeds = [int(s) for s in R.cfg("SKAB", "seeds", (0, 1, 2))]
    stride = int(R.cfg("SKAB", "stride_train", 1))
    bs, lr = int(R.cfg("SKAB", "batch_size", 128)), float(R.cfg("SKAB", "lr", 1e-3))

    ds = D.load_skab(win=win, stride_train=stride, calib_len=max(NCS))
    hz = ds["meta"]["sampling_hz"]
    seqs, trX, vaX = ds["test_seqs"], ds["train"].X, ds["val"].X
    print(f"[sens] SKAB train {trX.shape} val {vaX.shape} runs {len(seqs)} "
          f"device {dev} epochs {epochs} seeds {seeds}", flush=True)

    prev = R.load(OUT)
    rows = prev.get("rows", []) if prev else []
    done = {(r["model"], r["seed"]) for r in rows}
    for seed in seeds:
        for name in R.STATE_MODELS:
            if (name, seed) in done:
                continue
            print(f"{R.stamp()} --- {name} | seed {seed}", flush=True)
            val_sc, seq_sc, tr = R.fit_and_score(name, seed, trX, vaX, seqs, epochs, bs, lr, dev)
            for q in QS:
                thr_val = float(np.percentile(val_sc, q))
                glob = R.slim(R.evaluate_scores(seq_sc, seqs, thr_val, hz))
                for nc in NCS:
                    thr_run = []
                    for sc, ws in zip(seq_sc, seqs):
                        cal = np.asarray(ws.end_idx) < nc
                        thr_run.append(float(np.percentile(sc[cal], q)) if cal.sum() >= 10
                                       else thr_val)
                    run = R.slim(R.evaluate_scores(seq_sc, seqs, thr_run, hz))
                    rows.append({"model": name, "seed": seed, "q": q, "nc": nc,
                                 "n_calib_windows": int((np.asarray(seqs[0].end_idx) < nc).sum()),
                                 "global": glob, "run": run})
            R.dump({"qs": QS, "ncs": NCS, "win": win, "epochs": epochs, "seeds": seeds,
                    "device": dev, "rows": rows}, OUT)
            ref = [r for r in rows if r["model"] == name and r["seed"] == seed
                   and r["q"] == 99.0 and r["nc"] == 100][0]
            print(f"    q=99 Nc=100: F1 glob {ref['global']['f1']:.3f} run {ref['run']['f1']:.3f} "
                  f"| FAR glob {ref['global']['far']:.3f} run {ref['run']['far']:.3f}", flush=True)
    print("saved", R.dump({"qs": QS, "ncs": NCS, "win": win, "epochs": epochs, "seeds": seeds,
                           "device": dev, "rows": rows}, OUT))


if __name__ == "__main__":
    main()
