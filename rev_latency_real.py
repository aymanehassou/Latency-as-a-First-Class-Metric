"""Latency over distinct real test windows, with the
complete decision pipeline timed identically for every detector.

One pass = one process (run it five times with --pass-id 1..5, nothing else
running). For every detector two timings are taken in the same process:

  published : the forward-pass-only protocol of bench_latency.py. One fixed
              synthetic window, repeated. Neural models: 100 warm-up + 500 timed
              calls, the timed step being normalisation + forward pass.
              Classical detectors: 30 warm-up + 300 timed calls.
  full      : 500 DIFFERENT real test windows, each timed once after 100
              warm-up calls on other windows. The timed step is the whole
              decision for every detector: normalisation, forward pass or
              feature extraction, score (argmax, or reconstruction error),
              threshold comparison.

Latency does not depend on the learned weights for the neural models, so they
are instantiated untrained, as in the published pass. The tree ensembles and
PCA are fitted on real training windows because their cost depends on the fit.

Output: results/rev_latency_real_pass<K>.json
"""
from __future__ import annotations

import argparse
import pickle
import time

import numpy as np
import torch

import rev_common as R
from rev_common import D, L, M

N_WARM, N_ITER = 100, 500
TAU = 1.0  # any constant: the comparison costs the same


def pct(ts: np.ndarray) -> dict:
    return {"lat_mean_ms": float(ts.mean()), "lat_p50_ms": float(np.percentile(ts, 50)),
            "lat_p95_ms": float(np.percentile(ts, 95)), "lat_p99_ms": float(np.percentile(ts, 99)),
            "lat_max_ms": float(ts.max())}


def time_steps(step, windows) -> dict:
    """Warm up on the first N_WARM windows, then time one call per remaining window."""
    for w in windows[:N_WARM]:
        step(w)
    timed = windows[N_WARM: N_WARM + N_ITER]
    ts = np.empty(len(timed))
    for i, w in enumerate(timed):
        t0 = time.perf_counter()
        step(w)
        ts[i] = (time.perf_counter() - t0) * 1e3
    return pct(ts)


def time_fixed(step, window, n_warm=30, n_iter=300) -> dict:
    """Published protocol for the classical detectors: 30 warm-up calls, then
    300 timed calls on one fixed window (bench_latency.classical_latency)."""
    for _ in range(n_warm):
        step(window)
    ts = np.empty(n_iter)
    for i in range(n_iter):
        t0 = time.perf_counter()
        step(window)
        ts[i] = (time.perf_counter() - t0) * 1e3
    return pct(ts)


def pick(X: np.ndarray, n: int, rng) -> np.ndarray:
    idx = np.sort(rng.choice(len(X), size=min(n, len(X)), replace=len(X) < n))
    return np.stack([np.ascontiguousarray(X[i]) for i in idx]).astype(np.float32)


def rf_feats(X):
    return np.concatenate(
        [X.mean(1), X.std(1), X.min(1), X.max(1),
         np.percentile(X, 25, axis=1), np.percentile(X, 75, axis=1),
         np.median(np.abs(np.diff(X, axis=1)), axis=1),
         np.sqrt((X ** 2).mean(1)),
         np.abs(np.fft.rfft(X, axis=1)).argmax(1)], axis=1).astype(np.float32)


def if_feats(X):
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1),
                           np.abs(np.diff(X, axis=1)).mean(1)], axis=1)


@torch.no_grad()
def bench_neural(model, family, raw_real, mu, sd, win, ch):
    model.eval()
    # published protocol: one synthetic window, identity normalisation, forward only
    pub = L.measure_stream_latency(model, np.random.randn(win, ch).astype(np.float32),
                                   np.zeros(ch, dtype=np.float32), np.ones(ch, dtype=np.float32),
                                   n_warmup=N_WARM, n_iter=N_ITER)
    pub = {k: pub[k] for k in ("lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms",
                               "lat_max_ms")}
    if family == "clf":
        def step(raw):
            x = (raw - mu) / sd
            return int(model(torch.from_numpy(x).unsqueeze(0)).argmax(1))
    else:
        def step(raw):
            x = (raw - mu) / sd
            xt = torch.from_numpy(x).unsqueeze(0)
            return float(((model(xt) - xt) ** 2).mean()) > TAU
    full = time_steps(step, raw_real)
    Xb = np.random.randn(2048, win, ch).astype(np.float32)
    thr = L.measure_batch_throughput(model, Xb, batch=256, n_batches=8)
    return pub, full, thr


def bench_classical(score_pub, score_full, raw_real, mu, sd, win, ch):
    one = np.random.randn(1, win, ch).astype(np.float32)
    pub = time_fixed(lambda w: score_pub(w), one)

    def step(raw):
        x = ((raw - mu) / sd)[None]
        return score_full(x)
    full = time_steps(step, raw_real)
    Xb = np.random.randn(2048, win, ch).astype(np.float32)
    t0 = time.perf_counter()
    score_pub(Xb)
    thr = {"batch_throughput_wps": float(2048 / (time.perf_counter() - t0)), "batch_size": 2048}
    return pub, full, thr


def record(runs, cfg_name, model_name, win, ch, hz, params, size_kb, pub, full, thr):
    for d in (pub, full):
        d["stream_throughput_wps"] = float(1000.0 / d["lat_mean_ms"])
        d["realtime_margin"] = float(1000.0 / d["lat_p95_ms"] / hz)
    runs.append({"config": cfg_name, "model": model_name, "win": win, "channels": ch,
                 "sampling_hz": hz, "params": int(params), "size_kb": float(size_kb),
                 "published": pub, "full": full, **thr})
    print(f"{cfg_name:6s} {model_name:16s} published p50={pub['lat_p50_ms']:8.3f} ms   "
          f"full p50={full['lat_p50_ms']:8.3f} ms  ({full['lat_p50_ms']/pub['lat_p50_ms']:.2f}x)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass-id", type=int, default=1)
    args = ap.parse_args()

    L.set_single_thread()
    torch.manual_seed(0)
    np.random.seed(0)
    rng = np.random.RandomState(0)
    n_need = N_WARM + N_ITER
    runs, extra = [], {}

    # ---------------- WISDM (behaviour) ------------------------------------ #
    win = int(R.cfg("WISDM", "win", 128))
    st = int(R.cfg("WISDM", "stride", win // 2))
    ds = D.load_wisdm(win=win, stride_train=st, stride_test=st)
    meta = ds["test"].meta
    mu, sd, ch, hz = meta["mu"], meta["sd"], 3, 20.0
    raw = pick(ds["test"].X, n_need, rng) * sd + mu
    for name, ctor in M.CLASSIFIERS.items():
        model = ctor(ch, win, meta["n_classes"])
        pub, full, thr = bench_neural(model, "clf", raw, mu, sd, win, ch)
        record(runs, "WISDM", name, win, ch, hz, M.count_params(model), M.model_size_kb(model),
               pub, full, thr)
        del model

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=0,
                                 class_weight="balanced_subsample")
    clf.fit(rf_feats(ds["train"].X), ds["train"].y)
    nodes = int(sum(t.tree_.node_count for t in clf.estimators_))
    pub, full, thr = bench_classical(lambda X: clf.predict(rf_feats(X)),
                                     lambda X: int(clf.predict(rf_feats(X))[0]),
                                     raw, mu, sd, win, ch)
    extra["rf_nodes"] = nodes
    extra["rf_size_kb_estimate_40B_per_node"] = nodes * 40 / 1024.0
    extra["rf_pickle_kb"] = len(pickle.dumps(clf, protocol=pickle.HIGHEST_PROTOCOL)) / 1024.0
    extra["rf_max_depth"] = int(max(t.tree_.max_depth for t in clf.estimators_))
    record(runs, "WISDM", "RF-features", win, ch, hz, nodes, extra["rf_pickle_kb"], pub, full, thr)
    del clf, ds

    # ---------------- SKAB and SMD (state) --------------------------------- #
    state_cfgs = []
    w = int(R.cfg("SKAB", "win", 30))
    sk = D.load_skab(win=w, stride_train=int(R.cfg("SKAB", "stride_train", 1)), calib_len=100)
    pool = np.concatenate([s.X for s in sk["test_seqs"]])
    state_cfgs.append(("SKAB", w, sk["meta"], pick(pool, n_need, rng), sk["train"].X))
    del pool
    w = int(R.cfg("SMD", "win", 100))
    ent = D.smd_entities()[0]
    sm = D.load_smd_entity(ent, win=w, stride_train=int(R.cfg("SMD", "stride_train", 10)))
    state_cfgs.append(("SMD", w, sm["meta"], pick(sm["test_seqs"][0].X, n_need, rng),
                       sm["train"].X))
    extra["smd_entity_for_latency"] = ent

    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    for cfg_name, win, meta, real_norm, trX in state_cfgs:
        mu, sd, ch, hz = meta["mu"], meta["sd"], int(meta["n_channels"]), meta["sampling_hz"]
        raw = real_norm * sd + mu
        for name, ctor in M.RECONSTRUCTORS.items():
            model = ctor(ch, win)
            pub, full, thr = bench_neural(model, "ae", raw, mu, sd, win, ch)
            record(runs, cfg_name, name, win, ch, hz, M.count_params(model),
                   M.model_size_kb(model), pub, full, thr)
            del model

        Xn = np.random.randn(3000, win, ch).astype(np.float32)          # published fit
        Xr = pick(trX, 3000, rng)                                         # real fit
        for label, Xfit in (("pub", Xn), ("real", Xr)):
            pca = PCA(n_components=16, random_state=0).fit(Xfit.reshape(len(Xfit), -1))
            iso = IsolationForest(n_estimators=100, random_state=0, n_jobs=1).fit(if_feats(Xfit))
            if label == "pub":
                pca_pub, iso_pub = pca, iso
            else:
                pca_real, iso_real = pca, iso

        def pca_score(p):
            def f(X):
                flat = X.reshape(len(X), -1)
                return ((flat - p.inverse_transform(p.transform(flat))) ** 2).mean(1)
            return f

        pub, full, thr = bench_classical(pca_score(pca_pub),
                                         lambda X: float(pca_score(pca_real)(X)[0]) > TAU,
                                         raw, mu, sd, win, ch)
        npar = int(pca_real.components_.size + pca_real.mean_.size)
        record(runs, cfg_name, "PCA", win, ch, hz, npar, npar * 4 / 1024.0, pub, full, thr)

        pub, full, thr = bench_classical(lambda X: -iso_pub.score_samples(if_feats(X)),
                                         lambda X: float(-iso_real.score_samples(if_feats(X))[0]) > TAU,
                                         raw, mu, sd, win, ch)
        nodes = int(sum(e.tree_.node_count for e in iso_real.estimators_))
        record(runs, cfg_name, "IsolationForest", win, ch, hz, nodes,
               len(pickle.dumps(iso_real, protocol=pickle.HIGHEST_PROTOCOL)) / 1024.0,
               pub, full, thr)

    out = {"pass": args.pass_id, "n_warmup": N_WARM, "n_iter": N_ITER,
           "machine": {"threads": 1, "torch": torch.__version__}, "extra": extra, "runs": runs}
    print("saved", R.dump(out, "rev_latency_real_pass%d.json" % args.pass_id))


if __name__ == "__main__":
    main()
