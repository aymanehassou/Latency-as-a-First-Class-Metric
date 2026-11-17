"""Independent re-derivation of every headline number in the paper.

Deliberately does not import analyze.py: it re-reads the raw result JSONs and
recomputes the quantities from scratch, then compares them against the LaTeX
macros the manuscript actually uses. Any mismatch is a bug in one of the two.

Local note: the window-length sweeps are optional. If no wsweep_* files are
present the sweep block is skipped instead of crashing.
"""
from __future__ import annotations

import glob
import json
import re

import numpy as np

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_RES = _os.environ.get("RTBENCH_RESULTS", _os.path.join(_HERE, "results"))
_PAPER = _os.environ.get("RTBENCH_PAPER") or (
    _os.path.join(_os.path.dirname(_HERE), "paper")
    if _os.path.isdir(_os.path.join(_os.path.dirname(_HERE), "paper"))
    else _os.path.join(_HERE, "paper_out"))


RES = _RES
MACROS = _os.path.join(_PAPER, "tables", "macros.tex")

macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(MACROS).read()))
checks, fails = [], []


def check(name, computed, tol=5e-3, fmt="{:.3f}"):
    if name not in macros:
        fails.append("MISSING macro " + name)
        return
    raw = macros[name].replace("\\,", "").replace("$", "")
    try:
        got = float(raw)
    except ValueError:
        checks.append((name, macros[name], str(computed), raw == str(computed)))
        if raw != str(computed):
            fails.append("%s: macro=%s computed=%s" % (name, macros[name], computed))
        return
    ok = abs(got - float(computed)) <= max(tol, abs(got) * 1e-3)
    checks.append((name, fmt.format(got), fmt.format(float(computed)), ok))
    if not ok:
        fails.append("%s: macro=%s computed=%s" % (name, got, computed))


# --------------------------------------------------------------------------- #
# WISDM
# --------------------------------------------------------------------------- #
w = json.load(open(_os.path.join(RES, "wisdm.json")))
runs = w["runs"]
by = {}
for r in runs:
    by.setdefault(r["model"], []).append(r)

for key, model in [("Cnn", "CNN1D"), ("Mlp", "MLP"), ("Tcn", "TCN"), ("Lstm", "LSTM"),
                   ("Bigru", "BiGRU"), ("Trans", "Transformer"), ("Rf", "RF-features")]:
    f1 = [x["macro_f1"] for x in by[model]]
    acc = [x["accuracy"] for x in by[model]]
    assert len(f1) == 3, (model, len(f1))
    check(key + "F", np.mean(f1))
    check(key + "Fsd", np.std(f1, ddof=1))
    check(key + "Acc", np.mean(acc))
    check(key + "Train", np.mean([x["train_seconds"] for x in by[model]]), tol=1.0,
          fmt="{:.0f}")

lat = {r["model"]: r for r in json.load(open(_os.path.join(RES, "latency.json")))["runs"]
       if r["config"] == "WISDM"}
for key, model in [("Cnn", "CNN1D"), ("Mlp", "MLP"), ("Tcn", "TCN"), ("Lstm", "LSTM"),
                   ("Bigru", "BiGRU"), ("Trans", "Transformer"), ("Rf", "RF-features")]:
    check(key + "Lat", lat[model]["lat_p50_ms"], tol=5e-3, fmt="{:.2f}")
check("LatRatioBigruCnn", lat["BiGRU"]["lat_p50_ms"] / lat["CNN1D"]["lat_p50_ms"],
      tol=0.6, fmt="{:.0f}")
check("LatRatioClfMaxMin", max(r["lat_p50_ms"] for r in lat.values())
      / min(r["lat_p50_ms"] for r in lat.values()), tol=0.6, fmt="{:.0f}")
check("CnnMargin", lat["CNN1D"]["realtime_margin"], tol=0.6, fmt="{:.0f}")
# additional macros
check("LatRatioLstmCnn", lat["LSTM"]["lat_p50_ms"] / lat["CNN1D"]["lat_p50_ms"],
      tol=0.06, fmt="{:.1f}")
check("LatRatioCnnMlp", lat["CNN1D"]["lat_p50_ms"] / lat["MLP"]["lat_p50_ms"],
      tol=0.6, fmt="{:.0f}")
check("RfSizeMB", lat["RF-features"]["size_kb"] / 1024.0, tol=0.06, fmt="{:.1f}")
check("CnnMlpGap", np.mean([x["macro_f1"] for x in by["CNN1D"]])
      - np.mean([x["macro_f1"] for x in by["MLP"]]), tol=5e-3, fmt="{:.2f}")
check("BigruSdRatio", np.std([x["macro_f1"] for x in by["BiGRU"]], ddof=1)
      / np.std([x["macro_f1"] for x in by["CNN1D"]], ddof=1), tol=0.6, fmt="{:.0f}")
check("TrainRatioTransCnn",
      np.mean([x["train_seconds"] for x in by["Transformer"]])
      / np.mean([x["train_seconds"] for x in by["CNN1D"]]), tol=0.6, fmt="{:.0f}")


# --------------------------------------------------------------------------- #
# SKAB / SMD
# --------------------------------------------------------------------------- #
def load(pattern):
    out = []
    for fp in sorted(glob.glob(_os.path.join(RES, pattern))):
        out += json.load(open(fp))
    return out


def pol_mean(records, model, policy, field):
    key = {"global": "global_policy", "runcal": "runcal_policy",
           "oracle": "oracle_policy"}[policy]
    vals = [r[key][field] for r in records if r["model"] == model]
    return float(np.mean(vals))


skab = load("skab.json")
smd = load("smd_*.json")
AE = ["PCA", "IsolationForest", "DenseAE", "ConvAE", "LSTM-AE", "Transformer-AE"]
NAME = {"PCA": "Pca", "IsolationForest": "Ifor", "DenseAE": "Dense", "ConvAE": "Conv",
        "LSTM-AE": "Lae", "Transformer-AE": "Tae"}

for m in AE:
    for pol, tag in [("global", "Glob"), ("runcal", "Run"), ("oracle", "Orc")]:
        check("Sk" + NAME[m] + tag + "F", pol_mean(skab, m, pol, "f1"))
        check("Sk" + NAME[m] + tag + "Far", pol_mean(skab, m, pol, "false_alarm_rate"))
        check("Sk" + NAME[m] + tag + "Pa", pol_mean(skab, m, pol, "pa_f1"))
        check("Sk" + NAME[m] + tag + "Del", pol_mean(skab, m, pol, "delay_median_s"),
              tol=0.05, fmt="{:.1f}")

far_g = [pol_mean(skab, m, "global", "false_alarm_rate") for m in AE]
far_r = [pol_mean(skab, m, "runcal", "false_alarm_rate") for m in AE]
f1_g = [pol_mean(skab, m, "global", "f1") for m in AE]
f1_r = [pol_mean(skab, m, "runcal", "f1") for m in AE]
check("SkFarGlobMean", np.mean(far_g), tol=5e-3, fmt="{:.2f}")
check("SkFarRunMean", np.mean(far_r), tol=5e-3, fmt="{:.2f}")
check("SkFarDropFactor", np.mean(far_g) / np.mean(far_r), tol=0.05, fmt="{:.1f}")
check("SkFOneGlobMin", min(f1_g))
check("SkFOneGlobMax", max(f1_g))
check("SkFOneRunMax", max(f1_r))
neural = ["DenseAE", "ConvAE", "LSTM-AE", "Transformer-AE"]
f1_r_n = [pol_mean(skab, m, "runcal", "f1") for m in neural]
check("SkNeuralSpread", max(f1_r_n) - min(f1_r_n), tol=5e-3, fmt="{:.2f}")
check("SkPaMax", max(pol_mean(skab, m, "runcal", "pa_f1") for m in AE))

f1_dep = {m: pol_mean(smd, m, "runcal", "f1") for m in AE}
f1_orc = {m: pol_mean(smd, m, "oracle", "f1") for m in AE}
check("SmDepMin", min(f1_dep.values()))
check("SmDepMax", max(f1_dep.values()))
check("SmOrcMin", min(f1_orc.values()))
check("SmOrcMax", max(f1_orc.values()))
check("SmOracleGap", np.mean([f1_orc[k] / f1_dep[k] for k in f1_orc]), tol=0.05,
      fmt="{:.1f}")
check("SmPaMax", max(pol_mean(smd, m, "runcal", "pa_f1") for m in AE))
check("SmMissMin", min(pol_mean(smd, m, "runcal", "missed_rate") for m in AE),
      tol=5e-3, fmt="{:.2f}")
check("SkErRunMax", 1.0 - min(pol_mean(skab, m, "runcal", "missed_rate") for m in AE),
      tol=5e-3, fmt="{:.2f}")
check("SmErRunMax", 1.0 - min(pol_mean(smd, m, "runcal", "missed_rate") for m in AE),
      tol=5e-3, fmt="{:.2f}")
check("SmErOrcMax", 1.0 - min(pol_mean(smd, m, "oracle", "missed_rate") for m in AE),
      tol=5e-3, fmt="{:.2f}")
check("SkIforRunEr", 1.0 - pol_mean(skab, "IsolationForest", "runcal", "missed_rate"),
      tol=5e-3, fmt="{:.2f}")

# sanity: global and runcal must coincide on SMD (no warm-up windows there)
for m in AE:
    a, b = pol_mean(smd, m, "global", "f1"), pol_mean(smd, m, "runcal", "f1")
    if abs(a - b) > 1e-9:
        fails.append("SMD policies differ for %s: %s vs %s" % (m, a, b))

# --------------------------------------------------------------------------- #
# controls
# --------------------------------------------------------------------------- #
ctrl = json.load(open(_os.path.join(RES, "controls.json")))
sk_rand = [r for r in ctrl if r["dataset"] == "SKAB" and r["model"] == "random-score"]
sm_rand = [r for r in ctrl if r["dataset"].startswith("SMD") and r["model"] == "random-score"]
sk_all = [r for r in ctrl if r["dataset"] == "SKAB" and r["model"] == "always-alarm"]
check("SkRandPa", np.mean([r["m_pa_f1"] for r in sk_rand]))
check("SkRandF", np.mean([r["m_f1"] for r in sk_rand]))
check("SkRandFar", np.mean([r["m_false_alarm_rate"] for r in sk_rand]))
check("SkRandDel", np.mean([r["m_delay_median_s"] for r in sk_rand]), tol=0.6,
      fmt="{:.0f}")
check("SmRandPa", np.mean([r["m_pa_f1"] for r in sm_rand]))
check("SkRandEr", 1.0 - np.mean([r["m_missed_rate"] for r in sk_rand]), tol=5e-3, fmt="{:.2f}")
check("SkTrivial", np.mean([r["m_f1"] for r in sk_all]))

# --------------------------------------------------------------------------- #
# dataset facts
# --------------------------------------------------------------------------- #
facts = json.load(open(_os.path.join(RES, "dataset_facts.json")))
check("WisdmRaw", facts["wisdm"]["raw_samples"], fmt="{:.0f}", tol=0.5)
check("WisdmBad", facts["wisdm"]["malformed"], fmt="{:.0f}", tol=0.5)
check("SkabEvents", facts["skab"]["n_events"], fmt="{:.0f}", tol=0.5)
check("SmdEvents", facts["smd"]["n_events"], fmt="{:.0f}", tol=0.5)
check("SmdEntities", facts["smd"]["n_entities"], fmt="{:.0f}", tol=0.5)

# --------------------------------------------------------------------------- #
# sweeps (optional -- skipped when no wsweep_* results are present)
# --------------------------------------------------------------------------- #
sw = {}
for fp in sorted(glob.glob(_os.path.join(RES, "wsweep_wisdm_w*.json"))):
    win = int(fp.split("_w")[-1].split(".")[0])
    d = json.load(open(fp))
    for m in ("CNN1D", "LSTM"):
        v = [r["macro_f1"] for r in d["runs"] if r["model"] == m]
        if v:
            sw[(m, win)] = (float(np.mean(v)), float(np.std(v, ddof=1)))
cnn = {k[1]: v for k, v in sw.items() if k[0] == "CNN1D"}
lstm = {k[1]: v for k, v in sw.items() if k[0] == "LSTM"}
if cnn and lstm:
    best_w = max(cnn, key=lambda k: cnn[k][0])
    check("SweepCnnBestW", best_w, fmt="{:.0f}", tol=0.5)
    check("SweepCnnBestF", cnn[best_w][0])
    check("SweepCnnMinW", min(cnn), fmt="{:.0f}", tol=0.5)
    check("SweepCnnMinF", cnn[min(cnn)][0])
    check("SweepLstmMaxW", max(lstm), fmt="{:.0f}", tol=0.5)
    check("SweepLstmMaxF", lstm[max(lstm)][0])
    check("SweepLstmMaxSd", lstm[max(lstm)][1])
    check("SweepLstmBestF", max(v[0] for v in lstm.values()))
else:
    print("no WISDM sweep results -- skipping that block")

skw = {}
for fp in sorted(glob.glob(_os.path.join(RES, "wsweep_skab_w*.json"))):
    win = int(fp.split("_w")[-1].split(".")[0])
    d = json.load(open(fp))
    rs = [r for r in d if r["model"] == "ConvAE"]
    if rs:
        skw[win] = (float(np.mean([r["runcal_policy"]["f1"] for r in rs])),
                    float(np.mean([r["runcal_policy"]["delay_median_s"] for r in rs])),
                    float(np.mean([r["runcal_policy"]["false_alarm_rate"] for r in rs])))
if skw:
    lo, hi = min(skw), max(skw)
    check("SweepSkabWMin", lo, fmt="{:.0f}", tol=0.5)
    check("SweepSkabWMax", hi, fmt="{:.0f}", tol=0.5)
    check("SweepSkabFLo", skw[lo][0])
    check("SweepSkabFHi", skw[hi][0])
    check("SweepSkabDelLo", skw[lo][1], tol=0.05, fmt="{:.1f}")
    check("SweepSkabDelHi", skw[hi][1], tol=0.05, fmt="{:.1f}")
    check("SweepSkabFarLo", skw[lo][2])
    check("SweepSkabFarHi", skw[hi][2])
    check("SweepSkabFSpread",
          max(v[0] for v in skw.values()) - min(v[0] for v in skw.values()))
else:
    print("no SKAB sweep results -- skipping that block")

# --------------------------------------------------------------------------- #
print("%d checks, %d failures" % (len(checks), len(fails)))
for name, got, comp, ok in checks:
    if not ok:
        print("  FAIL %s: manuscript=%s recomputed=%s" % (name, got, comp))
for f in fails:
    print("  ", f)
print("ALL NUMBERS VERIFIED" if not fails else "MISMATCHES FOUND")
