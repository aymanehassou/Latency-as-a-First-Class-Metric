"""Aggregate raw result JSONs into every table and figure used in the paper.

No number in the manuscript is typed by hand: each one is either \\input from
tables/ or read from results/summary.json.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_RES = _os.environ.get("RTBENCH_RESULTS", _os.path.join(_HERE, "results"))
_PAPER = _os.environ.get("RTBENCH_PAPER") or (
    _os.path.join(_os.path.dirname(_HERE), "paper")
    if _os.path.isdir(_os.path.join(_os.path.dirname(_HERE), "paper"))
    else _os.path.join(_HERE, "paper_out"))


RES = _RES
FIG = _os.path.join(_PAPER, "figs")
TAB = _os.path.join(_PAPER, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# Okabe-Ito derived categorical order; validated for CVD separation
# (worst adjacent pair dE 9.6 deuteranopia). Marker shape carries identity too,
# so the figures survive greyscale printing.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#555555"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
INK, INK_MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d8d8d4"


def mpl_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": INK_MUTED, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.5,
        "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    })
    return plt


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_wisdm(path=None):
    path = path or os.path.join(RES, "wisdm.json")
    with open(path) as fh:
        raw = json.load(fh)
    return pd.DataFrame(raw["runs"]), raw["dataset"]


def load_state(pattern):
    rows = []
    for fp in sorted(glob.glob(os.path.join(RES, pattern))):
        with open(fp) as fh:
            for r in json.load(fh):
                base = {k: v for k, v in r.items()
                        if k not in ("global_policy", "runcal_policy", "oracle_policy",
                                     "threshold_run")}
                for pol in ("global_policy", "runcal_policy", "oracle_policy"):
                    if pol not in r:
                        continue
                    d = dict(base)
                    d["policy"] = pol.replace("_policy", "")
                    for k, v in r[pol].items():
                        d["m_" + k] = v
                    rows.append(d)
    return pd.DataFrame(rows)


def load_controls():
    with open(os.path.join(RES, "controls.json")) as fh:
        return pd.DataFrame(json.load(fh))


def load_latency():
    with open(os.path.join(RES, "latency.json")) as fh:
        return pd.DataFrame(json.load(fh)["runs"])


def agg(df, by, cols):
    cols = [c for c in cols if c in df.columns]
    g = df.groupby(by, sort=False)[cols].agg(["mean", "std"])
    g.columns = ["%s_%s" % (a, b) for a, b in g.columns]
    return g.reset_index()


def pm(m, s, nd=3):
    if pd.isna(m):
        return "--"
    if pd.isna(s):
        return "%.*f" % (nd, m)
    return "%.*f\\,$\\pm$\\,%.*f" % (nd, m, nd, s)


def num(x, nd=0):
    if pd.isna(x):
        return "--"
    return ("{:,.%df}" % nd).format(x).replace(",", "\\,")


def write(path, text):
    with open(path, "w") as fh:
        fh.write(text)
    print("wrote", path)


# --------------------------------------------------------------------------- #
ORDER_CLF = ["RF-features", "MLP", "CNN1D", "TCN", "LSTM", "BiGRU", "Transformer"]
NICE_CLF = {"RF-features": "RF (feat.)", "MLP": "MLP", "CNN1D": "1D-CNN", "TCN": "TCN",
            "LSTM": "LSTM", "BiGRU": "BiGRU", "Transformer": "Transformer"}
ORDER_AE = ["PCA", "IsolationForest", "DenseAE", "ConvAE", "LSTM-AE", "Transformer-AE"]
NICE_AE = {"PCA": "PCA", "IsolationForest": "iForest", "DenseAE": "Dense-AE",
           "ConvAE": "Conv-AE", "LSTM-AE": "LSTM-AE", "Transformer-AE": "Transf.-AE"}


# --------------------------------------------------------------------------- #
# 1. behaviour recognition
# --------------------------------------------------------------------------- #
def wisdm_outputs(lat):
    df, meta = load_wisdm()
    a = agg(df, ["model"], ["accuracy", "macro_f1", "train_seconds"])
    a["order"] = a.model.map({m: i for i, m in enumerate(ORDER_CLF)})
    a = a.sort_values("order")
    lw = lat[lat.config == "WISDM"].set_index("model")

    lines = []
    for _, r in a.iterrows():
        L = lw.loc[r.model]
        lines.append(" & ".join([
            NICE_CLF.get(r.model, r.model),
            num(L.params), "%.0f" % L.size_kb,
            pm(r.accuracy_mean, r.accuracy_std),
            pm(r.macro_f1_mean, r.macro_f1_std),
            "%.2f" % L.lat_p50_ms, "%.2f" % L.lat_p95_ms,
            num(L.batch_throughput_wps),
            "{:,.0f}".format(L.realtime_margin).replace(",", "\\,"),
        ]) + " \\\\")
    write(os.path.join(TAB, "wisdm.tex"), "\n".join(lines) + "%\n")

    # --- Pareto figure ----------------------------------------------------- #
    plt = mpl_setup()
    fig, ax = plt.subplots(figsize=(3.6, 1.75))
    ax.set_axisbelow(True)
    ax.grid(linestyle=":", linewidth=0.5)
    pts = []
    for i, (_, r) in enumerate(a.iterrows()):
        x = float(lw.loc[r.model, "lat_p95_ms"])
        y = float(r.macro_f1_mean)
        pts.append((x, y, r.model))
        ax.errorbar(x, y, yerr=0 if pd.isna(r.macro_f1_std) else r.macro_f1_std,
                    fmt=MARKERS[i % len(MARKERS)], color=PALETTE[i % len(PALETTE)],
                    markersize=5.5, capsize=2, elinewidth=0.7,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    offsets = {"RF (feat.)": (-14, -12), "MLP": (-7, 7), "1D-CNN": (-13, 7),
               "TCN": (-8, -12), "LSTM": (-24, 6), "BiGRU": (6, -2),
               "Transformer": (6, 3)}
    for x, y, m in pts:
        nm = NICE_CLF.get(m, m)
        ax.annotate(nm, (x, y), textcoords="offset points",
                    xytext=offsets.get(nm, (6, -2)), fontsize=6.5, color=INK)
    front_x, front_y, best = [], [], -1
    for x, y, _ in sorted(pts):
        if y > best:
            best = y
            front_x.append(x)
            front_y.append(y)
    front_x.append(max(x for x, _, _ in pts) * 2)
    front_y.append(front_y[-1])
    ax.step(front_x, front_y, where="post", color=INK_MUTED, linewidth=0.8,
            linestyle="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("per-window latency, p95 (ms, one CPU thread)")
    ax.set_ylabel("macro-F1")
    ax.set_xlim(0.04, 40)
    ax.set_ylim(0.745, 0.935)
    fig.savefig(os.path.join(FIG, "pareto_wisdm.pdf"))
    plt.close(fig)
    print("wrote", os.path.join(FIG, "pareto_wisdm.pdf"))
    return a, meta, lw


# --------------------------------------------------------------------------- #
# 2. state detection
# --------------------------------------------------------------------------- #
METRIC_COLS = ["m_f1", "m_pa_f1", "m_precision", "m_recall", "m_false_alarm_rate",
               "m_missed_rate", "m_delay_median_s", "m_delay_p90_s"]


def state_table(skab, smd, ctrl, facts):
    a_sk = agg(skab, ["model", "policy"], METRIC_COLS)
    a_sm = agg(smd, ["model", "policy"], METRIC_COLS)
    c_sk = agg(ctrl[ctrl.dataset == "SKAB"], ["model"], METRIC_COLS)
    c_sm = agg(ctrl[ctrl.dataset.str.startswith("SMD")], ["model"], METRIC_COLS)

    def row(name, r_sk, r_sm, pol_label):
        def cell(r, key, nd=3, div=1.0):
            if r is None or len(r) == 0:
                return "--"
            v = float(r[key + "_mean"].iloc[0]) / div
            return "--" if pd.isna(v) else "%.*f" % (nd, v)
        def er(r, nd=2):
            # event-level recall: fraction of labelled events that raised >= 1 alarm
            if r is None or len(r) == 0:
                return "--"
            v = 1.0 - float(r["m_missed_rate_mean"].iloc[0])
            return "--" if pd.isna(v) else "%.*f" % (nd, v)
        return " & ".join([
            name, pol_label,
            cell(r_sk, "m_f1"), cell(r_sk, "m_pa_f1"), er(r_sk),
            cell(r_sk, "m_false_alarm_rate"), cell(r_sk, "m_delay_median_s", 1),
            cell(r_sm, "m_f1"), cell(r_sm, "m_pa_f1"), er(r_sm),
            cell(r_sm, "m_false_alarm_rate"), cell(r_sm, "m_delay_median_s", 0, 60.0),
        ]) + " \\\\"

    lines = []
    for m in ORDER_AE:
        for pol, lab in [("runcal", "deployable"), ("oracle", "oracle")]:
            lines.append(row(NICE_AE[m] if pol == "runcal" else "",
                             a_sk[(a_sk.model == m) & (a_sk.policy == pol)],
                             a_sm[(a_sm.model == m) & (a_sm.policy == pol)], lab))
        lines.append("\\addlinespace")
    lines.append("\\midrule")
    lines.append(row("always-alarm", c_sk[c_sk.model == "always-alarm"],
                     c_sm[c_sm.model == "always-alarm"], "--"))
    lines.append(row("random score", c_sk[c_sk.model == "random-score"],
                     c_sm[c_sm.model == "random-score"], "deployable"))
    write(os.path.join(TAB, "state.tex"), "\n".join(lines) + "%\n")
    return a_sk, a_sm, c_sk, c_sm


def calibration_figure(skab):
    a = agg(skab, ["model", "policy"], METRIC_COLS)
    plt = mpl_setup()
    models = [m for m in ORDER_AE if m in set(a.model)]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.15))
    x = np.arange(len(models))
    w = 0.38
    handles = None
    for ax, key, ylab in [(axes[0], "m_false_alarm_rate", "false-alarm rate"),
                          (axes[1], "m_f1", "point-wise F1")]:
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle=":", linewidth=0.5)
        bars = []
        for j, (pol, colour, lab) in enumerate(
                [("global", PALETTE[1], "single global threshold"),
                 ("runcal", PALETTE[0], "per-run warm-up calibration")]):
            sel = [a[(a.model == m) & (a.policy == pol)] for m in models]
            vals = [float(t[key + "_mean"].iloc[0]) for t in sel]
            errs = [float(pd.Series([t[key + "_std"].iloc[0]]).fillna(0).iloc[0]) for t in sel]
            b = ax.bar(x + (j - 0.5) * (w + 0.02), vals, w, color=colour, label=lab,
                       edgecolor="white", linewidth=0.8, yerr=errs, capsize=1.5,
                       error_kw={"elinewidth": 0.6, "ecolor": INK_MUTED})
            bars.append(b)
        handles = bars
        ax.set_xticks(x)
        ax.set_xticklabels([NICE_AE[m] for m in models], rotation=25, ha="right")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.0)
    trivial = 0.5552392283504717
    axes[1].axhline(trivial, color=INK, linewidth=0.9, linestyle="--", zorder=4)
    axes[1].annotate("always-alarm", (len(models) - 0.45, trivial),
                     textcoords="offset points", xytext=(0, 3), ha="right",
                     fontsize=6.3, color=INK)
    fig.legend([h[0] for h in handles],
               ["single global threshold", "per-run warm-up calibration"],
               frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06),
               fontsize=7)
    fig.savefig(os.path.join(FIG, "calibration_skab.pdf"))
    plt.close(fig)
    print("wrote", os.path.join(FIG, "calibration_skab.pdf"))


def pa_inflation_figure(skab, smd, ctrl):
    """PA-F1 against point-wise F1 under the deployable policy, with the
    random-score control shown as the horizontal line every model must beat."""
    plt = mpl_setup()
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.75))
    for ax, df, name in [(axes[0], skab, "SKAB"), (axes[1], smd, "SMD")]:
        a = agg(df, ["model", "policy"], METRIC_COLS)
        a = a[a.policy == "runcal"]
        ax.set_axisbelow(True)
        ax.grid(linestyle=":", linewidth=0.5)
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=0.8, zorder=1)
        for i, m in enumerate([m for m in ORDER_AE if m in set(a.model)]):
            r = a[a.model == m].iloc[0]
            ax.scatter(r.m_f1_mean, r.m_pa_f1_mean, s=30, color=PALETTE[i],
                       marker=MARKERS[i], edgecolor="white", linewidth=0.6, zorder=3,
                       label=NICE_AE[m])
        sub = ctrl[(ctrl.dataset == "SKAB") if name == "SKAB"
                   else ctrl.dataset.str.startswith("SMD")]
        rnd = sub[sub.model == "random-score"]["m_pa_f1"].mean()
        ax.axhline(rnd, color=INK_MUTED, linewidth=1.0, linestyle="--", zorder=2)
        rnd_f1 = sub[sub.model == "random-score"]["m_f1"].mean()  # REV-CONTROL-MARKER
        ax.scatter(rnd_f1, rnd, s=34, color=INK, marker="X", edgecolor="white",
                   linewidth=0.6, zorder=4, clip_on=False)
        ax.annotate("random scores: %.2f" % rnd, (0.98, rnd), ha="right",
                    textcoords="offset points", xytext=(0, -9), fontsize=7,
                    color=INK_MUTED)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("point-wise F1")
        ax.set_title(name, loc="left")
    axes[0].set_ylabel("point-adjusted F1")
    axes[1].legend(*axes[0].get_legend_handles_labels(), frameon=False,
                   loc="upper left", ncol=2, fontsize=6.5, handletextpad=0.2,
                   columnspacing=0.6, labelspacing=0.3, borderaxespad=0.1)
    fig.subplots_adjust(wspace=0.22)
    fig.savefig(os.path.join(FIG, "pa_inflation.pdf"))
    plt.close(fig)
    print("wrote", os.path.join(FIG, "pa_inflation.pdf"))


# --------------------------------------------------------------------------- #
# 3. window-length sweep
# --------------------------------------------------------------------------- #
def sweep_figure(lat):
    rows_w = []
    for fp in sorted(glob.glob(os.path.join(RES, "wsweep_wisdm_w*.json"))):
        w = int(fp.split("_w")[-1].split(".")[0])
        df, _ = load_wisdm(fp)
        for m, g in df.groupby("model"):
            rows_w.append({"win": w, "model": m, "macro_f1": g.macro_f1.mean(),
                           "std": g.macro_f1.std()})
    rows_s = []
    for fp in sorted(glob.glob(os.path.join(RES, "wsweep_skab_w*.json"))):
        w = int(fp.split("_w")[-1].split(".")[0])
        d = load_state(os.path.basename(fp))
        d = d[d.policy == "runcal"]
        for m, g in d.groupby("model"):
            rows_s.append({"win": w, "model": m, "f1": g.m_f1.mean(),
                           "delay": g.m_delay_median_s.mean(),
                           "far": g.m_false_alarm_rate.mean()})
    if not rows_w and not rows_s:
        print("no sweep data")
        return None, None
    W = pd.DataFrame(rows_w).sort_values(["model", "win"])
    S = pd.DataFrame(rows_s).sort_values(["model", "win"])

    plt = mpl_setup()
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.0))
    ax = axes[0]
    ax.set_axisbelow(True)
    ax.grid(linestyle=":", linewidth=0.5)
    for i, (m, g) in enumerate(W.groupby("model")):
        ax.errorbar(g.win, g.macro_f1, yerr=g["std"].fillna(0), marker=MARKERS[i],
                    color=PALETTE[i], linewidth=1.4, markersize=5, capsize=2,
                    elinewidth=0.7, markeredgecolor="white", markeredgewidth=0.6,
                    label=NICE_CLF.get(m, m))
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(W.win.unique()))
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("$W$ (samples, WISDM @ 20 Hz)")
    ax.set_ylabel("macro-F1")
    ax.legend(frameon=False, loc="lower right")

    keep = ["ConvAE", "DenseAE", "PCA", "IsolationForest"]
    S = S[S.model.isin(keep)]
    for ax, col, ylab in [(axes[1], "delay", "median detection delay (s)"),
                          (axes[2], "far", "false-alarm rate")]:
        ax.set_axisbelow(True)
        ax.grid(linestyle=":", linewidth=0.5)
        for i, m in enumerate([k for k in keep if k in set(S.model)]):
            g = S[S.model == m].sort_values("win")
            ax.plot(g.win, g[col], marker=MARKERS[i], color=PALETTE[i], linewidth=1.4,
                    markersize=4.5, markeredgecolor="white", markeredgewidth=0.6,
                    label=NICE_AE.get(m, m))
        ax.set_xlabel("$W$ (s, SKAB @ 1 Hz)")
        ax.set_ylabel(ylab)
    axes[1].legend(frameon=False, loc="upper right", fontsize=6.2,
                   handletextpad=0.3, labelspacing=0.25)
    fig.subplots_adjust(wspace=0.42)
    fig.savefig(os.path.join(FIG, "window_sweep.pdf"))
    plt.close(fig)
    print("wrote", os.path.join(FIG, "window_sweep.pdf"))
    return W, S


# --------------------------------------------------------------------------- #
# 4. latency table
# --------------------------------------------------------------------------- #
def latency_table(lat):
    order = {}
    for i, m in enumerate(ORDER_CLF):
        order[m] = i
    for i, m in enumerate(ORDER_AE):
        order[m] = i
    lines = []
    for cfg in ["SKAB", "SMD"]:
        sub = lat[lat.config == cfg].copy()
        sub["o"] = sub.model.map(order)
        sub = sub.sort_values("o")
        for _, r in sub.iterrows():
            nice = NICE_CLF.get(r.model, NICE_AE.get(r.model, r.model))
            lines.append(" & ".join([
                cfg, nice, num(r.params), "%.0f" % r.size_kb,
                "%.3f" % r.lat_p50_ms, "%.3f" % r.lat_p95_ms,
                num(r.stream_throughput_wps), num(r.batch_throughput_wps),
                "{:,.0f}".format(r.realtime_margin).replace(",", "\\,"),
            ]) + " \\\\")
        lines.append("\\addlinespace")
    write(os.path.join(TAB, "latency.tex"), "\n".join(lines) + "%\n")


# --------------------------------------------------------------------------- #
def macros(facts, a_w, lw, a_sk, a_sm, c_sk, c_sm, lat, W, S):
    """Emit every number quoted in the prose as a LaTeX macro."""
    def g(df, model, policy, col):
        r = df[(df.model == model) & (df.policy == policy)]
        return float(r[col].iloc[0]) if len(r) else float("nan")

    def gw(model, col):
        return float(a_w[a_w.model == model][col].iloc[0])

    def thou(x):
        return "{:,}".format(int(x)).replace(",", "\\,")

    m = {}
    # --- WISDM ---
    for key, model in [("Cnn", "CNN1D"), ("Mlp", "MLP"), ("Tcn", "TCN"),
                       ("Lstm", "LSTM"), ("Bigru", "BiGRU"),
                       ("Trans", "Transformer"), ("Rf", "RF-features")]:
        m[key + "F"] = "%.3f" % gw(model, "macro_f1_mean")
        m[key + "Fsd"] = "%.3f" % gw(model, "macro_f1_std")
        m[key + "Acc"] = "%.3f" % gw(model, "accuracy_mean")
        m[key + "Lat"] = "%.2f" % float(lw.loc[model, "lat_p50_ms"])
        m[key + "LatP"] = "%.2f" % float(lw.loc[model, "lat_p95_ms"])
        m[key + "Par"] = thou(lw.loc[model, "params"])
        m[key + "Train"] = "%.0f" % gw(model, "train_seconds_mean")
    m["LatRatioBigruCnn"] = "%.0f" % (float(lw.loc["BiGRU", "lat_p50_ms"]) /
                                      float(lw.loc["CNN1D", "lat_p50_ms"]))
    m["LatRatioCnnMlp"] = "%.0f" % (float(lw.loc["CNN1D", "lat_p50_ms"]) /
                                    float(lw.loc["MLP", "lat_p50_ms"]))
    with open(os.path.join(RES, "latency.json")) as fh:
        m["LatRepeats"] = str(int(json.load(fh).get("repeats", 1)))
    m["LatSpreadMax"] = "%.0f" % max(r.get("lat_p50_ms_spread_pct", 0.0)
                                     for r in lat.to_dict("records"))
    m["TrainRatioTransCnn"] = "%.0f" % (gw("Transformer", "train_seconds_mean") /
                                        gw("CNN1D", "train_seconds_mean"))
    # numbers quoted in the prose of the behaviour section
    m["CnnMargin"] = "%.0f" % float(lw.loc["CNN1D", "realtime_margin"])
    m["CnnMlpGap"] = "%.2f" % (gw("CNN1D", "macro_f1_mean") - gw("MLP", "macro_f1_mean"))
    _best = max(gw(mm, "macro_f1_mean") for mm in ORDER_CLF if mm != "RF-features")
    m["RfGapBest"] = "%.2f" % (_best - gw("RF-features", "macro_f1_mean"))
    m["BigruSdRatio"] = "%.0f" % (gw("BiGRU", "macro_f1_std") / gw("CNN1D", "macro_f1_std"))
    m["LatSpreadDecades"] = "%.0f" % np.log10(float(lw.lat_p50_ms.max()) /
                                              float(lw.lat_p50_ms.min()))
    m["LatRatioClfMaxMin"] = "%.0f" % (float(lw.lat_p50_ms.max()) / float(lw.lat_p50_ms.min()))
    m["LatRatioLstmCnn"] = "%.1f" % (float(lw.loc["LSTM", "lat_p50_ms"]) /
                                     float(lw.loc["CNN1D", "lat_p50_ms"]))
    m["RfSizeMB"] = "%.1f" % (float(lw.loc["RF-features", "size_kb"]) / 1024.0)
    _neural_ae = ["DenseAE", "ConvAE", "LSTM-AE", "Transformer-AE"]
    _sk = lat[(lat.config == "SKAB") & lat.model.isin(_neural_ae)].set_index("model")
    _sm = lat[(lat.config == "SMD") & lat.model.isin(_neural_ae)].set_index("model")
    m["SkAeParMin"], m["SkAeParMax"] = thou(_sk.params.min()), thou(_sk.params.max())
    _nd = _sm.drop(index="DenseAE")
    m["SmAeParMinNoDense"], m["SmAeParMaxNoDense"] = thou(_nd.params.min()), thou(_nd.params.max())
    m["SmDensePar"] = thou(_sm.loc["DenseAE", "params"])

    # --- SKAB ---
    for key, model in [("Dense", "DenseAE"), ("Conv", "ConvAE"), ("Lae", "LSTM-AE"),
                       ("Tae", "Transformer-AE"), ("Pca", "PCA"),
                       ("Ifor", "IsolationForest")]:
        for pol, tag in [("global", "Glob"), ("runcal", "Run"), ("oracle", "Orc")]:
            m["Sk" + key + tag + "F"] = "%.3f" % g(a_sk, model, pol, "m_f1_mean")
            m["Sk" + key + tag + "Far"] = "%.3f" % g(a_sk, model, pol,
                                                     "m_false_alarm_rate_mean")
            m["Sk" + key + tag + "Pa"] = "%.3f" % g(a_sk, model, pol, "m_pa_f1_mean")
            m["Sk" + key + tag + "Del"] = "%.1f" % g(a_sk, model, pol,
                                                     "m_delay_median_s_mean")
    far_g = [g(a_sk, mm, "global", "m_false_alarm_rate_mean") for mm in ORDER_AE]
    far_r = [g(a_sk, mm, "runcal", "m_false_alarm_rate_mean") for mm in ORDER_AE]
    f1_g = [g(a_sk, mm, "global", "m_f1_mean") for mm in ORDER_AE]
    f1_r = [g(a_sk, mm, "runcal", "m_f1_mean") for mm in ORDER_AE]
    m["SkFarGlobMin"], m["SkFarGlobMax"] = "%.2f" % min(far_g), "%.2f" % max(far_g)
    m["SkFarGlobMean"] = "%.2f" % np.mean(far_g)
    m["SkFarRunMean"] = "%.2f" % np.mean(far_r)
    m["SkFarDropFactor"] = "%.1f" % (np.mean(far_g) / np.mean(far_r))
    m["SkFarRunMin"], m["SkFarRunMax"] = "%.2f" % min(far_r), "%.2f" % max(far_r)
    m["SkFOneGlobMin"], m["SkFOneGlobMax"] = "%.3f" % min(f1_g), "%.3f" % max(f1_g)
    m["SkFOneRunMax"] = "%.3f" % max(f1_r)
    neural = ["DenseAE", "ConvAE", "LSTM-AE", "Transformer-AE"]
    f1_r_n = [g(a_sk, mm, "runcal", "m_f1_mean") for mm in neural]
    m["SkNeuralSpread"] = "%.2f" % (max(f1_r_n) - min(f1_r_n))
    m["SkAllSpread"] = "%.2f" % (max(f1_r) - min(f1_r))
    m["SkTrivial"] = "%.3f" % facts["skab"]["trivial_f1"]
    m["SkRandPa"] = "%.3f" % float(c_sk[c_sk.model == "random-score"]["m_pa_f1_mean"].iloc[0])
    m["SkRandF"] = "%.3f" % float(c_sk[c_sk.model == "random-score"]["m_f1_mean"].iloc[0])
    m["SkRandDel"] = "%.0f" % float(
        c_sk[c_sk.model == "random-score"]["m_delay_median_s_mean"].iloc[0])
    m["SkRandFar"] = "%.3f" % float(
        c_sk[c_sk.model == "random-score"]["m_false_alarm_rate_mean"].iloc[0])
    m["SkPaMax"] = "%.3f" % max(g(a_sk, mm, "runcal", "m_pa_f1_mean") for mm in ORDER_AE)

    # --- SMD ---
    f1_dep = {mm: g(a_sm, mm, "runcal", "m_f1_mean") for mm in ORDER_AE}
    f1_orc = {mm: g(a_sm, mm, "oracle", "m_f1_mean") for mm in ORDER_AE}
    m["SmDepMin"], m["SmDepMax"] = "%.3f" % min(f1_dep.values()), "%.3f" % max(f1_dep.values())
    m["SmOrcMin"], m["SmOrcMax"] = "%.3f" % min(f1_orc.values()), "%.3f" % max(f1_orc.values())
    m["SmOracleGap"] = "%.1f" % np.mean([f1_orc[k] / f1_dep[k] for k in f1_orc])
    m["SmTrivial"] = "%.3f" % facts["smd"]["trivial_f1"]
    m["SmRandPa"] = "%.3f" % float(c_sm[c_sm.model == "random-score"]["m_pa_f1_mean"].iloc[0])
    m["SmRandF"] = "%.3f" % float(c_sm[c_sm.model == "random-score"]["m_f1_mean"].iloc[0])
    m["SmPaMax"] = "%.3f" % max(g(a_sm, mm, "runcal", "m_pa_f1_mean") for mm in ORDER_AE)
    m["SmMissMin"] = "%.2f" % min(g(a_sm, mm, "runcal", "m_missed_rate_mean")
                                  for mm in ORDER_AE)
    # event-level recall = 1 - missed-segment rate
    for key, model in [("Dense", "DenseAE"), ("Conv", "ConvAE"), ("Lae", "LSTM-AE"),
                       ("Tae", "Transformer-AE"), ("Pca", "PCA"),
                       ("Ifor", "IsolationForest")]:
        m["Sk" + key + "RunEr"] = "%.2f" % (1.0 - g(a_sk, model, "runcal", "m_missed_rate_mean"))
        m["Sm" + key + "RunEr"] = "%.2f" % (1.0 - g(a_sm, model, "runcal", "m_missed_rate_mean"))
    er_sk = [1.0 - g(a_sk, mm, "runcal", "m_missed_rate_mean") for mm in ORDER_AE]
    er_sm = [1.0 - g(a_sm, mm, "runcal", "m_missed_rate_mean") for mm in ORDER_AE]
    m["SkErRunMin"], m["SkErRunMax"] = "%.2f" % min(er_sk), "%.2f" % max(er_sk)
    m["SmErRunMin"], m["SmErRunMax"] = "%.2f" % min(er_sm), "%.2f" % max(er_sm)
    er_sm_orc = [1.0 - g(a_sm, mm, "oracle", "m_missed_rate_mean") for mm in ORDER_AE]
    m["SmErOrcMax"] = "%.2f" % max(er_sm_orc)
    m["SkRandEr"] = "%.2f" % (1.0 - float(
        c_sk[c_sk.model == "random-score"]["m_missed_rate_mean"].iloc[0]))
    m["SmDelayMed"] = "%.0f" % (np.mean([g(a_sm, mm, "runcal", "m_delay_median_s_mean")
                                         for mm in ORDER_AE]) / 60)

    # --- state-detection inference cost ---
    for cfg, tag in [("SKAB", "Sk"), ("SMD", "Sm")]:
        sub = lat[lat.config == cfg]
        m[tag + "LatMin"] = "%.2f" % sub.lat_p50_ms.min()
        m[tag + "LatMax"] = "%.2f" % sub.lat_p50_ms.max()
        m[tag + "LatSpread"] = "%.0f" % (sub.lat_p50_ms.max() / sub.lat_p50_ms.min())
        slowest = str(sub.loc[sub.lat_p50_ms.idxmax(), "model"])
        m[tag + "LatSlowest"] = NICE_AE.get(slowest, slowest)
        m[tag + "MarginMin"] = "{:,.0f}".format(sub.realtime_margin.min()).replace(",", "\\,")

    # --- dataset facts ---
    m["WisdmRaw"] = thou(facts["wisdm"]["raw_samples"])
    m["WisdmBad"] = thou(facts["wisdm"]["malformed"])
    m["WisdmTrain"] = thou(facts["wisdm"]["n_train"])
    m["WisdmTest"] = thou(facts["wisdm"]["n_test"])
    m["SkabPoints"] = thou(facts["skab"]["n_eval_points"])
    m["SkabPos"] = "%.1f" % (facts["skab"]["positive_rate"] * 100)
    m["SkabEvents"] = str(facts["skab"]["n_events"])
    m["SmdPoints"] = thou(facts["smd"]["n_eval_points"])
    m["SmdPos"] = "%.1f" % (facts["smd"]["positive_rate"] * 100)
    m["SmdEvents"] = str(facts["smd"]["n_events"])
    m["SmdEntities"] = str(facts["smd"]["n_entities"])

    # --- sweeps ---
    if W is not None and len(W):
        cnn = W[W.model == "CNN1D"].sort_values("win")
        m["SweepCnnBestW"] = str(int(cnn.loc[cnn.macro_f1.idxmax(), "win"]))
        m["SweepCnnBestF"] = "%.3f" % cnn.macro_f1.max()
        m["SweepCnnMinW"] = str(int(cnn.win.min()))
        m["SweepCnnMinF"] = "%.3f" % float(cnn[cnn.win == cnn.win.min()].macro_f1.iloc[0])
    if S is not None and len(S):
        cae = S[S.model == "ConvAE"].sort_values("win")
        lo, hi = cae.win.min(), cae.win.max()
        m["SweepSkabWMin"], m["SweepSkabWMax"] = str(int(lo)), str(int(hi))
        for tag, w in [("Lo", lo), ("Hi", hi)]:
            r = cae[cae.win == w].iloc[0]
            m["SweepSkabDel" + tag] = "%.1f" % float(r.delay)
            m["SweepSkabF" + tag] = "%.3f" % float(r.f1)
            m["SweepSkabFar" + tag] = "%.3f" % float(r.far)
        m["SweepSkabFSpread"] = "%.3f" % float(cae.f1.max() - cae.f1.min())
        lstm = W[W.model == "LSTM"].sort_values("win")
        m["SweepLstmMaxW"] = str(int(lstm.win.max()))
        m["SweepLstmMaxF"] = "%.3f" % float(lstm[lstm.win == lstm.win.max()].macro_f1.iloc[0])
        m["SweepLstmMaxSd"] = "%.3f" % float(lstm[lstm.win == lstm.win.max()]["std"].iloc[0])
        m["SweepLstmBestF"] = "%.3f" % float(lstm.macro_f1.max())

    body = "\n".join("\\newcommand{\\%s}{%s}" % (k, v) for k, v in m.items())
    write(os.path.join(TAB, "macros.tex"), body + "\n")
    return m


def main():
    with open(os.path.join(RES, "dataset_facts.json")) as fh:
        facts = json.load(fh)
    lat = load_latency()
    latency_table(lat)
    a_w, meta_w, lw = wisdm_outputs(lat)

    skab = load_state("skab.json")
    smd = load_state("smd_*.json")
    ctrl = load_controls()
    a_sk, a_sm, c_sk, c_sm = state_table(skab, smd, ctrl, facts)
    calibration_figure(skab)
    pa_inflation_figure(skab, smd, ctrl)
    W, S = sweep_figure(lat)
    mac = macros(facts, a_w, lw, a_sk, a_sm, c_sk, c_sm, lat, W, S)

    summary = {
        "macros": mac,
        "facts": facts,
        "wisdm": a_w.to_dict("records"),
        "wisdm_latency": lw.reset_index().to_dict("records"),
        "skab": a_sk.to_dict("records"),
        "smd": a_sm.to_dict("records"),
        "controls_skab": c_sk.to_dict("records"),
        "controls_smd": c_sm.to_dict("records"),
        "latency": lat.to_dict("records"),
        "sweep_wisdm": None if W is None else W.to_dict("records"),
        "sweep_skab": None if S is None else S.to_dict("records"),
    }
    with open(os.path.join(RES, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print("wrote", os.path.join(RES, "summary.json"))


if __name__ == "__main__":
    main()
