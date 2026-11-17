"""Generate the dataset-summary table and the dataset facts used in the text."""
from __future__ import annotations

import json
import os

import numpy as np

import config  # noqa: F401  -- sets RTBENCH_DATA / RESULTS / PAPER
from rtbench import data as D
from rtbench import metrics as MT

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.environ.get("RTBENCH_RESULTS", os.path.join(_HERE, "results"))
_PAPER = os.environ.get("RTBENCH_PAPER") or os.path.join(_HERE, "paper_out")
TAB = os.path.join(_PAPER, "tables")
os.makedirs(TAB, exist_ok=True)
os.makedirs(_RES, exist_ok=True)
facts = {}

z = np.load(os.path.join(_HERE, "cache_wisdm_w128_s64.npz"), allow_pickle=True)
meta = z["meta"].item()
w_counts = {s: int(len(z[s + "_y"])) for s in ["train", "val", "test"]}
raw = D._read_wisdm_raw()
facts["wisdm"] = {
    "raw_samples": int(len(raw)), "malformed": int(raw.attrs["malformed_lines"]),
    "subjects": 36, "classes": 6, "channels": 3, "hz": 20.0, "win": 128,
    "win_seconds": 128 / 20.0, "stride": 64,
    "n_train": w_counts["train"], "n_val": w_counts["val"], "n_test": w_counts["test"],
    "test_class_counts": np.bincount(z["test_y"], minlength=6).tolist(),
}
del raw

sk = D.load_skab(win=30, stride_train=1, calib_len=100)
ev_true, n_seg = [], 0
for s in sk["test_seqs"]:
    keep = ~np.asarray(s.meta["is_calib"])
    y = s.y.astype(int)[keep]
    ev_true.append(y)
    n_seg += len(MT.find_segments(y))
y_sk = np.concatenate(ev_true)
facts["skab"] = {
    "channels": 8, "hz": 1.0, "win": 30, "win_seconds": 30.0, "calib_len": 100,
    "n_train_windows": int(len(sk["train"])), "n_val_windows": int(len(sk["val"])),
    "n_runs": len(sk["test_seqs"]), "n_eval_points": int(len(y_sk)),
    "positive_rate": float(y_sk.mean()), "n_events": int(n_seg),
    "trivial_f1": float(MT.state_scores(y_sk, np.ones_like(y_sk), 1.0)["f1"]),
}
del sk

ents = D.smd_entities()
n_tr = n_pts = n_seg = 0
pos = []
for e in ents:
    ds = D.load_smd_entity(e, win=100, stride_train=10)
    t = ds["test_seqs"][0]
    y = t.y.astype(int)
    n_tr += len(ds["train"])
    n_pts += len(y)
    n_seg += len(MT.find_segments(y))
    pos.append(y)
    del ds
y_smd = np.concatenate(pos)
facts["smd"] = {
    "channels": 38, "hz": 1 / 60.0, "win": 100, "win_seconds": 6000.0,
    "n_entities": len(ents), "n_train_windows": int(n_tr),
    "n_eval_points": int(n_pts), "positive_rate": float(y_smd.mean()),
    "n_events": int(n_seg),
    "trivial_f1": float(MT.state_scores(y_smd, np.ones_like(y_smd), 1 / 60.0)["f1"]),
}

rows = [
    ("WISDM v1.1", "wearable IMU", "behaviour (6)", 3, "20 Hz", "128 / 6.4 s",
     "{:,}".format(facts["wisdm"]["n_train"]),
     "{:,} w".format(facts["wisdm"]["n_test"]), "--", "--"),
    ("SKAB", "industrial rig", "state (bin.)", 8, "1 Hz", "30 / 30 s",
     "{:,}".format(facts["skab"]["n_train_windows"]),
     "{:,} p".format(facts["skab"]["n_eval_points"]),
     "{:.1f}\\%".format(facts["skab"]["positive_rate"] * 100),
     str(facts["skab"]["n_events"])),
    ("SMD", "server fleet", "state (bin.)", 38, "1/60 Hz", "100 / 100 min",
     "{:,}".format(facts["smd"]["n_train_windows"]),
     "{:,} p".format(facts["smd"]["n_eval_points"]),
     "{:.1f}\\%".format(facts["smd"]["positive_rate"] * 100),
     str(facts["smd"]["n_events"])),
]


def cell(c):
    t = str(c)
    return t.replace(",", "\\,") if t.replace(",", "").isdigit() else t


body = "\n".join(" & ".join(cell(c) for c in r) + " \\\\" for r in rows)

table = r"""\begin{table}
\caption{The three domains. Windows advance with stride 1 at evaluation time in
both state-detection domains, so positive rates and event counts are expressed
in sampling periods. Nominal warm-up samples are excluded from the SKAB counts}
\label{tab:datasets}
\begin{center}
\scriptsize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{llllllrrrr}
\toprule
Dataset & Domain & Task & $C$ & $f_s$ & $W$ & Train & Evaluated & Pos. & Ev. \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{center}
\end{table}
"""

with open(os.path.join(TAB, "datasets_table.tex"), "w") as fh:
    fh.write(table)
with open(os.path.join(_RES, "dataset_facts.json"), "w") as fh:
    json.dump(facts, fh, indent=2)
print(json.dumps(facts, indent=2))
print("wrote", os.path.join(TAB, "datasets_table.tex"))
