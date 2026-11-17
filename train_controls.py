"""The two degenerate controls -- not models, but the numbers that make the
state-detection results readable.

**always-alarm** flags every sample. It has zero detection delay, misses
nothing, and its F1 is the floor any useful detector must clear. On SKAB,
where 38% of the samples are anomalous, that floor is 0.555 -- higher than
several published detectors score. Quoting an F1 on an imbalanced-but-not-rare
benchmark without this number next to it tells the reader nothing.

**random score** draws i.i.d. uniform scores and pushes them through exactly
the same calibration procedure as the learned detectors. Following Kim et al.
(AAAI 2022), this is the control that exposes how much of a point-adjusted
score is an artefact of the scoring convention rather than a property of the
model: on SKAB these random scores reach a point-adjusted F1 above 0.97 while
their point-wise F1 is 0.04.

Neither control trains anything, so this runs in a few minutes.

    python train_controls.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import config
from rtbench import data as D
from engine.state import evaluate_scores

OUT = os.path.join(config.RESULTS_DIR, "controls.json")


def controls_for(ds, tag, q, seeds=(0, 1, 2)):
    seqs = ds["test_seqs"]
    hz = ds["meta"]["sampling_hz"]
    rows = []

    # ---- always-alarm: one threshold below every possible score ------------ #
    inf_sc = [np.full(len(s.y), 1.0) for s in seqs]
    res = evaluate_scores(inf_sc, seqs, 0.0, hz)
    row = {"dataset": tag, "model": "always-alarm", "seed": 0, "policy": "none"}
    row.update({"m_%s" % k: v for k, v in res.items()})
    rows.append(row)

    # ---- random scores, calibrated exactly like a learned detector --------- #
    for seed in seeds:
        rng = np.random.default_rng(seed)
        sc = [rng.random(len(s.y)) for s in seqs]
        thr_run = []
        for s, ws in zip(sc, seqs):
            cal = np.asarray(ws.meta.get("is_calib", np.zeros(len(s), dtype=bool)))
            thr_run.append(float(np.percentile(s[cal], q)) if cal.sum() >= 10
                           else float(np.percentile(s, q)))
        res = evaluate_scores(sc, seqs, thr_run, hz)
        row = {"dataset": tag, "model": "random-score", "seed": seed, "policy": "runcal"}
        row.update({"m_%s" % k: v for k, v in res.items()})
        rows.append(row)
    return rows


def main():
    config.banner("degenerate controls", "always-alarm and random score")
    rows = []

    ds = D.load_skab(win=config.SKAB["win"], stride_train=config.SKAB["stride_train"],
                     calib_len=config.SKAB["calib_len"])
    rows += controls_for(ds, "SKAB", q=config.SKAB["val_quantile"])
    del ds

    ents = D.smd_entities()
    for i, e in enumerate(ents, 1):
        print("-- SMD entity %2d/%d  %s" % (i, len(ents), e), flush=True)
        ds = D.load_smd_entity(e, win=config.SMD["win"],
                               stride_train=config.SMD["stride_train"])
        rows += controls_for(ds, "SMD/%s" % e, q=config.SMD["val_quantile"])
        del ds

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(rows, fh, indent=2)

    print("")
    for r in rows:
        if r["dataset"] == "SKAB":
            print("   %-13s seed %d | point F1 %.3f | PA-F1 %.3f | FAR %.3f | delay %.1f s"
                  % (r["model"], r["seed"], r["m_f1"], r["m_pa_f1"],
                     r["m_false_alarm_rate"], r["m_delay_median_s"]))
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
