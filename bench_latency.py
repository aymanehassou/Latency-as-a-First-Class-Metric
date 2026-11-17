"""Dedicated latency pass.

Inference cost does not depend on the learned weights, only on the architecture
and the input shape, so latency is measured in a separate, quiescent pass rather
than alongside training. The whole pass is repeated and the median across passes
reported: a single pass is internally stable to a few per cent, but a pass can
land on a throttled or contended host.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import config  # noqa: F401  -- sets RTBENCH_DATA / RESULTS / PAPER
from rtbench import models as M
from rtbench import latency as L

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.environ.get("RTBENCH_RESULTS", os.path.join(_HERE, "results"))
OUT = os.environ.get("RTBENCH_LATENCY_OUT", os.path.join(_RES, "latency.json"))

CONFIGS = {
    "WISDM": dict(win=128, ch=3, k=6, family="clf", hz=20.0),
    "SKAB": dict(win=30, ch=8, k=None, family="ae", hz=1.0),
    "SMD": dict(win=100, ch=38, k=None, family="ae", hz=1 / 60.0),
}


def measure_model(model, win, ch, n_iter=500):
    mu = np.zeros(ch, dtype=np.float32)
    sd = np.ones(ch, dtype=np.float32)
    raw = np.random.randn(win, ch).astype(np.float32)
    lat = L.measure_stream_latency(model, raw, mu, sd, n_warmup=100, n_iter=n_iter)
    X = np.random.randn(2048, win, ch).astype(np.float32)
    thr = L.measure_batch_throughput(model, X, batch=256, n_batches=8)
    out = dict(lat)
    out.update(thr)
    out["params"] = M.count_params(model)
    out["size_kb"] = M.model_size_kb(model)
    return out


def classical_latency(score_fn, win, ch, n_iter=300):
    one = np.random.randn(1, win, ch).astype(np.float32)
    for _ in range(30):
        score_fn(one)
    ts = np.empty(n_iter)
    for i in range(n_iter):
        t0 = time.perf_counter()
        score_fn(one)
        ts[i] = (time.perf_counter() - t0) * 1e3
    X = np.random.randn(2048, win, ch).astype(np.float32)
    t0 = time.perf_counter()
    score_fn(X)
    batch_wps = 2048 / (time.perf_counter() - t0)
    return {"lat_mean_ms": float(ts.mean()), "lat_p50_ms": float(np.percentile(ts, 50)),
            "lat_p95_ms": float(np.percentile(ts, 95)),
            "lat_p99_ms": float(np.percentile(ts, 99)), "lat_max_ms": float(ts.max()),
            "stream_throughput_wps": float(1000.0 / ts.mean()),
            "batch_throughput_wps": float(batch_wps), "batch_size": 2048}


def one_pass():
    L.set_single_thread()
    torch.manual_seed(0)
    np.random.seed(0)
    out = {"machine": {"threads": 1, "torch": torch.__version__,
                       "cpu_count": os.cpu_count()}, "runs": []}

    for cfg_name, cfg in CONFIGS.items():
        win, ch = cfg["win"], cfg["ch"]
        zoo = M.CLASSIFIERS if cfg["family"] == "clf" else M.RECONSTRUCTORS
        for mname, ctor in zoo.items():
            model = ctor(ch, win, cfg["k"]) if cfg["family"] == "clf" else ctor(ch, win)
            r = measure_model(model, win, ch)
            r.update({"config": cfg_name, "model": mname, "win": win, "channels": ch,
                      "sampling_hz": cfg["hz"],
                      "realtime_margin": float(1000.0 / r["lat_p95_ms"] / cfg["hz"])})
            out["runs"].append(r)
            print("%-6s %-16s p50=%8.3f ms  p95=%8.3f  params=%8d"
                  % (cfg_name, mname, r["lat_p50_ms"], r["lat_p95_ms"], r["params"]),
                  flush=True)
            del model

        if cfg["family"] == "clf":
            from sklearn.ensemble import RandomForestClassifier
            z = np.load(os.path.join(_HERE, "cache_wisdm_w128_s64.npz"), allow_pickle=True)
            Xtr, ytr = z["train_X"], z["train_y"]

            def feats(X):
                return np.concatenate(
                    [X.mean(1), X.std(1), X.min(1), X.max(1),
                     np.percentile(X, 25, axis=1), np.percentile(X, 75, axis=1),
                     np.median(np.abs(np.diff(X, axis=1)), axis=1),
                     np.sqrt((X ** 2).mean(1)),
                     np.abs(np.fft.rfft(X, axis=1)).argmax(1)], axis=1).astype(np.float32)

            clf = RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=0,
                                         class_weight="balanced_subsample")
            clf.fit(feats(Xtr), ytr)
            r = classical_latency(lambda X: clf.predict(feats(X)), win, ch)
            nodes = int(sum(t.tree_.node_count for t in clf.estimators_))
            r.update({"config": cfg_name, "model": "RF-features", "win": win,
                      "channels": ch, "sampling_hz": cfg["hz"], "params": nodes,
                      "size_kb": nodes * 40 / 1024.0,
                      "realtime_margin": float(1000.0 / r["lat_p95_ms"] / cfg["hz"])})
            out["runs"].append(r)
            print("%-6s %-16s p50=%8.3f ms" % (cfg_name, "RF-features", r["lat_p50_ms"]),
                  flush=True)
        else:
            from sklearn.decomposition import PCA
            from sklearn.ensemble import IsolationForest
            Xn = np.random.randn(3000, win, ch).astype(np.float32)
            pca = PCA(n_components=16, random_state=0).fit(Xn.reshape(len(Xn), -1))

            def pca_score(X):
                f = X.reshape(len(X), -1)
                return ((f - pca.inverse_transform(pca.transform(f))) ** 2).mean(1)

            r = classical_latency(pca_score, win, ch)
            r.update({"config": cfg_name, "model": "PCA", "win": win, "channels": ch,
                      "sampling_hz": cfg["hz"],
                      "params": int(pca.components_.size + pca.mean_.size),
                      "size_kb": (pca.components_.size + pca.mean_.size) * 4 / 1024.0,
                      "realtime_margin": float(1000.0 / r["lat_p95_ms"] / cfg["hz"])})
            out["runs"].append(r)
            print("%-6s %-16s p50=%8.3f ms" % (cfg_name, "PCA", r["lat_p50_ms"]), flush=True)

            def ifeats(X):
                return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1),
                                       np.abs(np.diff(X, axis=1)).mean(1)], axis=1)

            iso = IsolationForest(n_estimators=100, random_state=0, n_jobs=1)
            iso.fit(ifeats(Xn))
            r = classical_latency(lambda X: -iso.score_samples(ifeats(X)), win, ch)
            nodes = int(sum(e.tree_.node_count for e in iso.estimators_))
            r.update({"config": cfg_name, "model": "IsolationForest", "win": win,
                      "channels": ch, "sampling_hz": cfg["hz"], "params": nodes,
                      "size_kb": nodes * 40 / 1024.0,
                      "realtime_margin": float(1000.0 / r["lat_p95_ms"] / cfg["hz"])})
            out["runs"].append(r)
            print("%-6s %-16s p50=%8.3f ms" % (cfg_name, "IsolationForest", r["lat_p50_ms"]),
                  flush=True)
    return out


AGG_KEYS = ["lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms", "lat_max_ms",
            "stream_throughput_wps", "batch_throughput_wps", "realtime_margin"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    passes = []
    for i in range(args.repeats):
        print("### pass %d/%d" % (i + 1, args.repeats), flush=True)
        passes.append(one_pass())

    base = passes[0]
    merged = []
    for r in base["runs"]:
        key = (r["config"], r["model"])
        vals = {k: [] for k in AGG_KEYS}
        for p in passes:
            rr = next(x for x in p["runs"] if (x["config"], x["model"]) == key)
            for k in AGG_KEYS:
                if k in rr:
                    vals[k].append(float(rr[k]))
        rec = dict(r)
        for k, v in vals.items():
            if v:
                rec[k] = float(np.median(v))
        p50s = vals["lat_p50_ms"]
        rec["lat_p50_ms_min"] = float(min(p50s))
        rec["lat_p50_ms_max"] = float(max(p50s))
        rec["lat_p50_ms_spread_pct"] = float(100.0 * (max(p50s) - min(p50s)) / np.median(p50s))
        merged.append(rec)

    out = {"machine": base["machine"], "repeats": args.repeats, "runs": merged}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    worst = max(merged, key=lambda r: r["lat_p50_ms_spread_pct"])
    print("saved %s  (median of %d passes; worst inter-pass spread %.1f%% on %s/%s)"
          % (OUT, args.repeats, worst["lat_p50_ms_spread_pct"], worst["config"],
             worst["model"]))


if __name__ == "__main__":
    main()
