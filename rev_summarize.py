"""Collect the supplementary runs into numbers the paper can cite.

Reads   results/rev_sens_skab.json, rev_seeds_smd.json, rev_latency_real_pass*.json,
        rev_env.json
Writes  results/rev_summary.json          every aggregate, with the inputs it came from
        results/latency_real.json         'full' latency in the schema of latency.json
        paper_out/tables/rev_macros.tex   LaTeX macros (no number is typed by hand)
        paper_out/tables/rev_sens.tex     rows of the sensitivity table
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

import rev_common as R

NEURAL = ["DenseAE", "ConvAE", "LSTM-AE", "Transformer-AE"]
ALL = NEURAL + ["PCA", "IsolationForest"]
macros, summary = {}, {}


def f(x, nd=3):
    return ("%." + str(nd) + "f") % float(x)


def tex_escape(s) -> str:
    out = str(s)
    for a, b in (("\\", "/"), ("_", "\\_"), ("%", "\\%"), ("&", "\\&"), ("#", "\\#")):
        out = out.replace(a, b)
    return out


# --------------------------------------------------------------------------- #
def sensitivity():
    d = R.load("rev_sens_skab.json")
    if not d:
        print("[sum] no sensitivity results")
        return
    rows = d["rows"]
    qs, ncs = [float(q) for q in d["qs"]], [int(n) for n in d["ncs"]]
    models = [m for m in ALL if any(r["model"] == m for r in rows)]

    def seed_mean(model, q, nc, pol, key):
        v = [r[pol][key] for r in rows if r["model"] == model and r["q"] == q and r["nc"] == nc]
        return float(np.mean(v))

    grid, lines = {}, []
    for q in qs:
        far_glob = float(np.mean([seed_mean(m, q, ncs[-1], "global", "far") for m in models]))
        f_glob_max = float(np.max([seed_mean(m, q, ncs[-1], "global", "f1") for m in models]))
        cells = {}
        for nc in ncs:
            far_run = float(np.mean([seed_mean(m, q, nc, "run", "far") for m in models]))
            f_run = {m: seed_mean(m, q, nc, "run", "f1") for m in models}
            er_run = {m: seed_mean(m, q, nc, "run", "er") for m in models}
            neural = [f_run[m] for m in models if m in NEURAL]
            cells[nc] = {"far_run_mean": far_run, "drop_factor": far_glob / far_run,
                         "best_f1_run": max(f_run.values()),
                         "best_model": max(f_run, key=f_run.get),
                         "neural_f1_spread": (max(neural) - min(neural)) if neural else None,
                         "er_min": min(er_run.values()), "er_max": max(er_run.values())}
        grid[q] = {"far_global_mean": far_glob, "best_f1_global": f_glob_max, "per_nc": cells}
        lines.append("%s & %s & %s & %s \\\\" % (
            ("%g" % q), f(far_glob, 2),
            " & ".join(f(cells[nc]["far_run_mean"], 2) for nc in ncs),
            " & ".join(f(cells[nc]["best_f1_run"]) for nc in ncs)))
    summary["sensitivity"] = {"qs": qs, "ncs": ncs, "models": models, "grid": grid,
                              "n_seeds": len({r["seed"] for r in rows})}

    allc = [grid[q]["per_nc"][nc] for q in qs for nc in ncs]
    macros["SensQs"] = "/".join("%g" % q for q in qs)
    macros["SensNcs"] = "/".join(str(n) for n in ncs)
    macros["SensFarRunMin"] = f(min(c["far_run_mean"] for c in allc), 2)
    macros["SensFarRunMax"] = f(max(c["far_run_mean"] for c in allc), 2)
    macros["SensFarGlobMin"] = f(min(grid[q]["far_global_mean"] for q in qs), 2)
    macros["SensFarGlobMax"] = f(max(grid[q]["far_global_mean"] for q in qs), 2)
    macros["SensDropMin"] = f(min(c["drop_factor"] for c in allc), 1)
    macros["SensDropMax"] = f(max(c["drop_factor"] for c in allc), 1)
    macros["SensBestFMin"] = f(min(c["best_f1_run"] for c in allc))
    macros["SensBestFMax"] = f(max(c["best_f1_run"] for c in allc))
    macros["SensErMin"] = f(min(c["er_min"] for c in allc), 2)
    macros["SensBestGlobFMax"] = f(max(grid[q]["best_f1_global"] for q in qs))
    if 99.0 in grid and 100 in grid[99.0]["per_nc"]:
        ref = grid[99.0]["per_nc"][100]
        macros["SensRefFarRun"] = f(ref["far_run_mean"], 2)
        macros["SensRefFarGlob"] = f(grid[99.0]["far_global_mean"], 2)
        macros["SensRefBestF"] = f(ref["best_f1_run"])
    os.makedirs(os.path.join(R.PAPER, "tables"), exist_ok=True)
    with open(os.path.join(R.PAPER, "tables", "rev_sens.tex"), "w") as fh:
        fh.write("\n".join(lines) + "%\n")
    print("[sum] sensitivity grid (mean per-run FAR / best per-run F1):")
    for q in qs:
        print("   q=%-5g global FAR %.2f | " % (q, grid[q]["far_global_mean"]) + "  ".join(
            "Nc=%d: %.2f / %.3f" % (nc, grid[q]["per_nc"][nc]["far_run_mean"],
                                    grid[q]["per_nc"][nc]["best_f1_run"]) for nc in ncs))


# --------------------------------------------------------------------------- #
def smd_seeds():
    d = R.load("rev_seeds_smd.json")
    if not d:
        print("[sum] no SMD multi-seed results")
        return
    rows = d["rows"]
    ents = sorted({r["entity"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    models = [m for m in ALL if any(r["model"] == m for r in rows)]

    def macro_avg(model, seed, pol, key):
        v = [r[pol][key] for r in rows if r["model"] == model and r["seed"] == seed]
        return float(np.mean(v)) if len(v) == len(ents) else float("nan")

    per_model = {}
    for m in models:
        rec = {}
        for pol in ("deployable", "oracle"):
            for key in ("f1", "er", "far", "pa_f1"):
                v = np.array([macro_avg(m, s, pol, key) for s in seeds])
                rec["%s_%s" % (pol, key)] = {"per_seed": v.tolist(), "mean": float(v.mean()),
                                             "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
        per_model[m] = rec
    gap = [float(np.mean([macro_avg(m, s, "oracle", "f1") / macro_avg(m, s, "deployable", "f1")
                          for m in models])) for s in seeds]
    er_orc = [max(macro_avg(m, s, "oracle", "er") for m in models) for s in seeds]
    er_dep = [max(macro_avg(m, s, "deployable", "er") for m in models) for s in seeds]
    complete = not any(np.isnan(x) for m in models for x in per_model[m]["deployable_f1"]["per_seed"])
    summary["smd_seeds"] = {"entities": ents, "seeds": seeds, "models": models,
                            "complete": complete, "sd_ddof": 1, "per_model": per_model,
                            "oracle_gap_per_seed": gap, "er_oracle_max_per_seed": er_orc,
                            "er_deployable_max_per_seed": er_dep}
    macros["SmSeedEntities"] = str(len(ents))
    macros["SmSeedSeeds"] = str(len(seeds))
    macros["SmSeedDepSdMax"] = f(max(per_model[m]["deployable_f1"]["sd"] for m in models))
    macros["SmSeedOrcSdMax"] = f(max(per_model[m]["oracle_f1"]["sd"] for m in models))
    macros["SmSeedDepMin"] = f(min(per_model[m]["deployable_f1"]["mean"] for m in models))
    macros["SmSeedDepMax"] = f(max(per_model[m]["deployable_f1"]["mean"] for m in models))
    macros["SmSeedOrcMin"] = f(min(per_model[m]["oracle_f1"]["mean"] for m in models))
    macros["SmSeedOrcMax"] = f(max(per_model[m]["oracle_f1"]["mean"] for m in models))
    macros["SmSeedGapMin"] = f(min(gap), 1)
    macros["SmSeedGapMax"] = f(max(gap), 1)
    macros["SmSeedErOrcMin"] = f(min(er_orc), 2)
    macros["SmSeedErOrcMax"] = f(max(er_orc), 2)
    macros["SmSeedErDepMin"] = f(min(er_dep), 2)
    macros["SmSeedErDepMax"] = f(max(er_dep), 2)
    print("[sum] SMD seeds (%d entities, %d seeds, complete=%s):" % (len(ents), len(seeds), complete))
    for m in models:
        a, b = per_model[m]["deployable_f1"], per_model[m]["oracle_f1"]
        print("   %-15s deployable F1 %.3f +- %.3f | oracle F1 %.3f +- %.3f"
              % (m, a["mean"], a["sd"], b["mean"], b["sd"]))
    print("   oracle/deployable multiplier per seed:", ["%.2f" % g for g in gap])


# --------------------------------------------------------------------------- #
KEYS = ["lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms", "lat_max_ms",
        "stream_throughput_wps", "realtime_margin"]


def latency():
    files = sorted(glob.glob(os.path.join(R.RESULTS, "rev_latency_real_pass*.json")))
    if not files:
        print("[sum] no latency passes")
        return
    passes = [json.load(open(p)) for p in files]
    base = passes[0]
    merged, compare = [], []
    for r in base["runs"]:
        key = (r["config"], r["model"])
        same = [next(x for x in p["runs"] if (x["config"], x["model"]) == key) for p in passes]
        rec = {k: r[k] for k in ("config", "model", "win", "channels", "sampling_hz", "params",
                                 "size_kb", "batch_size")}
        rec["batch_throughput_wps"] = float(np.median([x["batch_throughput_wps"] for x in same]))
        out = {}
        for var in ("published", "full"):
            agg = {k: float(np.median([x[var][k] for x in same])) for k in KEYS}
            p50 = [x[var]["lat_p50_ms"] for x in same]
            agg["lat_p50_ms_min"], agg["lat_p50_ms_max"] = float(min(p50)), float(max(p50))
            agg["lat_p50_ms_spread_pct"] = float(100.0 * (max(p50) - min(p50)) / np.median(p50))
            out[var] = agg
        merged.append({**rec, **out["full"]})
        compare.append({"config": r["config"], "model": r["model"],
                        "published_p50_ms": out["published"]["lat_p50_ms"],
                        "full_p50_ms": out["full"]["lat_p50_ms"],
                        "published_p95_ms": out["published"]["lat_p95_ms"],
                        "full_p95_ms": out["full"]["lat_p95_ms"],
                        "ratio_p50": out["full"]["lat_p50_ms"] / out["published"]["lat_p50_ms"],
                        "abs_diff_ms": out["full"]["lat_p50_ms"] - out["published"]["lat_p50_ms"],
                        "full_margin": out["full"]["realtime_margin"],
                        "full_spread_pct": out["full"]["lat_p50_ms_spread_pct"],
                        "published_spread_pct": out["published"]["lat_p50_ms_spread_pct"]})
    R.dump({"machine": base["machine"], "repeats": len(passes), "protocol":
            "500 distinct real test windows, full decision pipeline", "runs": merged},
           "latency_real.json")
    extra = passes[0].get("extra", {})
    summary["latency"] = {"n_passes": len(passes), "compare": compare, "extra": extra}

    def p50(cfg, model, var="full"):
        c = next(x for x in compare if x["config"] == cfg and x["model"] == model)
        return c["%s_p50_ms" % var]

    wis = [c for c in compare if c["config"] == "WISDM"]
    dev = [abs(c["ratio_p50"] - 1.0) * 100 for c in compare]
    macros["LatRealPasses"] = str(len(passes))
    macros["LatRealDevMaxPct"] = f(max(dev), 0)
    macros["LatRealDevMedPct"] = f(float(np.median(dev)), 0)
    macros["LatRealAbsDiffMaxMs"] = f(max(abs(c["abs_diff_ms"]) for c in compare), 2)
    macros["LatRealSpreadMaxPct"] = f(max(c["full_spread_pct"] for c in compare), 0)
    for var, tag in (("full", "Real"), ("published", "Pub")):
        macros["Lat%sCnn" % tag] = f(p50("WISDM", "CNN1D", var), 2)
        macros["Lat%sMlp" % tag] = f(p50("WISDM", "MLP", var), 2)
        macros["Lat%sLstm" % tag] = f(p50("WISDM", "LSTM", var), 2)
        macros["Lat%sBigru" % tag] = f(p50("WISDM", "BiGRU", var), 2)
        macros["Lat%sRf" % tag] = f(p50("WISDM", "RF-features", var), 2)
        macros["Lat%sRatioBigruCnn" % tag] = f(p50("WISDM", "BiGRU", var) / p50("WISDM", "CNN1D", var), 0)
        macros["Lat%sRatioLstmCnn" % tag] = f(p50("WISDM", "LSTM", var) / p50("WISDM", "CNN1D", var), 1)
        macros["Lat%sRatioCnnMlp" % tag] = f(p50("WISDM", "CNN1D", var) / p50("WISDM", "MLP", var), 0)
        v = [c["%s_p50_ms" % var] for c in wis]
        macros["Lat%sRatioMaxMin" % tag] = f(max(v) / min(v), 0)
    order_full = [c["model"] for c in sorted(wis, key=lambda c: c["full_p50_ms"])]
    order_pub = [c["model"] for c in sorted(wis, key=lambda c: c["published_p50_ms"])]
    summary["latency"]["wisdm_order_full"] = order_full
    summary["latency"]["wisdm_order_published"] = order_pub
    summary["latency"]["wisdm_order_unchanged"] = order_full == order_pub
    # the same forward-pass-only protocol, measured in an earlier session and
    # again in this one: how much the machine itself moved
    for cand in ("latency_forward_only.json", "latency.json"):
        prev = R.load(cand)
        if prev and "protocol" not in prev:
            old_p50 = {(x["config"], x["model"]): x["lat_p50_ms"] for x in prev["runs"]}
            chg = [100.0 * (c["published_p50_ms"] / old_p50[(c["config"], c["model"])] - 1.0)
                   for c in compare if (c["config"], c["model"]) in old_p50]
            summary["latency"]["session_change_pct"] = chg
            macros["LatSessionMedPct"] = f(abs(float(np.median(chg))), 0)
            macros["LatSessionMinPct"] = f(min(abs(x) for x in chg), 0)
            macros["LatSessionMaxPct"] = f(max(abs(x) for x in chg), 0)
            summary["latency"]["session_all_faster"] = bool(all(x < 0 for x in chg))
            w = {x["model"]: x["lat_p50_ms"] for x in prev["runs"] if x["config"] == "WISDM"}
            macros["LatSubmittedRatioBigruCnn"] = f(w["BiGRU"] / w["CNN1D"], 0)
            break
    ae = [c for c in compare if c["config"] != "WISDM"
          and c["model"] not in ("PCA", "IsolationForest")]
    clf = [c for c in wis if c["model"] != "RF-features"]
    cla = [c for c in compare if c["model"] in ("PCA", "IsolationForest", "RF-features")]
    macros["LatStepAeMaxPct"] = f(max(100 * (c["ratio_p50"] - 1) for c in ae), 0)
    macros["LatStepAeMinPct"] = f(min(100 * (c["ratio_p50"] - 1) for c in ae), 0)
    macros["LatStepClfMaxPct"] = f(max(abs(100 * (c["ratio_p50"] - 1)) for c in clf), 0)
    macros["LatStepClaMaxPct"] = f(max(abs(100 * (c["ratio_p50"] - 1)) for c in cla), 0)
    if "rf_pickle_kb" in extra:
        macros["RfPickleKB"] = f(extra["rf_pickle_kb"], 0)
        macros["RfPickleMB"] = f(extra["rf_pickle_kb"] / 1024.0, 1)
        macros["RfNodesRev"] = str(extra["rf_nodes"])
    print("[sum] latency, median of %d passes (p50 ms, published protocol -> full pipeline on "
          "real windows):" % len(passes))
    for c in compare:
        print("   %-6s %-16s %8.3f -> %8.3f  (%.2fx, spread %.0f%%)" % (
            c["config"], c["model"], c["published_p50_ms"], c["full_p50_ms"], c["ratio_p50"],
            c["full_spread_pct"]))
    print("   WISDM ordering unchanged:", order_full == order_pub)


# --------------------------------------------------------------------------- #
def environment():
    env = R.load("rev_env.json")
    if not env:
        print("[sum] no environment record")
        return
    summary["environment"] = env
    if "power_scheme" in env:
        macros["EnvPowerScheme"] = tex_escape(env["power_scheme"])
    if "on_ac_power" in env and env["on_ac_power"] is not None:
        macros["EnvPower"] = "mains power" if env["on_ac_power"] else "battery"
    if "priority_class" in env:
        macros["EnvPriority"] = tex_escape(env["priority_class"])
    if "affinity_pinned" in env:
        macros["EnvAffinity"] = ("pinned to %d of %d logical cores" % (
            env["affinity_cores_allowed"], env["affinity_cores_system"])
            if env["affinity_pinned"] else
            "not pinned (%d logical cores visible)" % env["affinity_cores_system"])
    macros["EnvTimer"] = tex_escape("%s (%s)" % (env["timer"]["function"],
                                                 env["timer"]["implementation"]))


def main():
    sensitivity()
    smd_seeds()
    latency()
    environment()
    R.dump({"macros": macros, **summary}, "rev_summary.json")
    tdir = os.path.join(R.PAPER, "tables")
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, "rev_macros.tex"), "w") as fh:
        fh.write("% generated by rev_summarize.py -- do not edit\n")
        for k in sorted(macros):
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, macros[k]))
    print("[sum] %d macros -> %s" % (len(macros), os.path.join(tdir, "rev_macros.tex")))


if __name__ == "__main__":
    main()
