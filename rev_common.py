"""Shared helpers for the supplementary experiments (sensitivity to q and N_c,
SMD seed check, whole-decision latency). Run them with run_revision.bat.

The training and scoring functions below are the ones behind the main
state-track results (engine/state.py), copied unchanged, so the supplementary
numbers are produced by the same recipe.
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# config.py points rtbench at the data and results folders: import it first.
CFG = None
try:
    import config as CFG  # noqa: E402
except Exception as exc:  # pragma: no cover
    print("[rev] config.py not importable (%s); falling back to defaults" % exc)

if "RTBENCH_DATA" not in os.environ:
    cands = [getattr(CFG, "DATA_DIR", None), os.path.join(HERE, "data")]
    for cand in cands:
        if cand and os.path.isdir(cand):
            os.environ["RTBENCH_DATA"] = cand
            break

RESULTS = os.environ.get("RTBENCH_RESULTS") or os.path.join(HERE, "results")
PAPER = os.environ.get("RTBENCH_PAPER") or os.path.join(HERE, "paper_out")
os.makedirs(RESULTS, exist_ok=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from rtbench import data as D  # noqa: E402
from rtbench import models as M  # noqa: E402
from rtbench import metrics as MT  # noqa: E402
from rtbench import latency as L  # noqa: E402


def cfg(section: str, key: str, default):
    """Read config.<SECTION>[key] if the project defines it, else the default
    used for the published run."""
    d = getattr(CFG, section, None) if CFG is not None else None
    if isinstance(d, dict) and key in d:
        return d[key]
    return default


def device() -> str:
    want = getattr(CFG, "DEVICE", None) if CFG is not None else None
    want = want or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(want).startswith("cuda") and not torch.cuda.is_available():
        print("[rev] CUDA requested but not available; training on CPU")
        return "cpu"
    return str(want)


def threads() -> int:
    return int(getattr(CFG, "THREADS", 4) if CFG is not None else 4)


def dump(obj, name: str) -> str:
    path = os.path.join(RESULTS, name)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)
    return path


def load(name: str, default=None):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return default
    with open(path) as fh:
        return json.load(fh)


def stamp() -> str:
    return time.strftime("%H:%M:%S")


# --------------------------------------------------------------------------- #
# deep reconstructors (unchanged from the published run)
# --------------------------------------------------------------------------- #
def train_ae(name: str, train_X: np.ndarray, val_X: np.ndarray, seed: int,
             epochs: int, bs: int, lr: float, verbose=False, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    t, c = train_X.shape[1], train_X.shape[2]
    model = M.RECONSTRUCTORS[name](c, t).to(dev)
    Xtr = torch.from_numpy(train_X).to(dev)
    Xva = torch.from_numpy(val_X).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.MSELoss()
    n = len(Xtr)
    best, best_state, patience, ep = float("inf"), None, 0, 0
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i: i + bs]
            opt.zero_grad()
            loss = crit(model(Xtr[idx]), Xtr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([float(crit(model(Xva[i: i + 512]), Xva[i: i + 512]))
                                for i in range(0, len(Xva), 512)]))
        if verbose:
            print(f"   ep{ep+1:02d} train={tot/n:.5f} val={vl:.5f}", flush=True)
        if vl < best - 1e-6:
            best, patience = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 5:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"train_seconds": time.perf_counter() - t0, "epochs_run": ep + 1,
                   "val_recon_mse": best}


def as_batches(X: np.ndarray, bs: int):
    for i in range(0, len(X), bs):
        yield i, np.ascontiguousarray(X[i: i + bs])


@torch.no_grad()
def ae_scores(model, X: np.ndarray, bs: int = 512, device="cpu") -> np.ndarray:
    model.eval()
    dev = torch.device(device)
    out = np.empty(len(X), dtype=np.float64)
    for i, chunk in as_batches(X, bs):
        xb = torch.from_numpy(chunk).to(dev)
        err = ((model(xb) - xb) ** 2).mean(dim=(1, 2))
        out[i: i + len(chunk)] = err.cpu().numpy()
    return out


def classical_scores(algo, X: np.ndarray, bs: int = 2048) -> np.ndarray:
    out = np.empty(len(X), dtype=np.float64)
    for i, chunk in as_batches(X, bs):
        out[i: i + len(chunk)] = algo.score(chunk)
    return out


# --------------------------------------------------------------------------- #
# classical detectors (unchanged from the published run)
# --------------------------------------------------------------------------- #
class PCARecon:
    name = "PCA"

    def __init__(self, k: int = 16, seed: int = 0):
        from sklearn.decomposition import PCA
        self.k, self.seed, self.pca = k, seed, None
        self._PCA = PCA

    def fit(self, X):
        n, t, c = X.shape
        self.t, self.c = t, c
        self.pca = self._PCA(n_components=min(self.k, t * c, n), random_state=self.seed)
        t0 = time.perf_counter()
        self.pca.fit(X.reshape(n, -1))
        return {"train_seconds": time.perf_counter() - t0,
                "params": int(self.pca.components_.size + self.pca.mean_.size)}

    def score(self, X):
        flat = X.reshape(len(X), -1)
        rec = self.pca.inverse_transform(self.pca.transform(flat))
        return ((flat - rec) ** 2).mean(1)


class IForest:
    name = "IsolationForest"

    def __init__(self, seed: int = 0, n_jobs: int = 2):
        from sklearn.ensemble import IsolationForest as IF
        self._IF, self.seed, self.m, self.n_jobs = IF, seed, None, n_jobs

    @staticmethod
    def _feats(X):
        return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1),
                               np.abs(np.diff(X, axis=1)).mean(1)], axis=1)

    def fit(self, X):
        self.m = self._IF(n_estimators=100, random_state=self.seed, n_jobs=self.n_jobs)
        t0 = time.perf_counter()
        self.m.fit(self._feats(X))
        return {"train_seconds": time.perf_counter() - t0,
                "params": int(sum(e.tree_.node_count for e in self.m.estimators_))}

    def score(self, X):
        return -self.m.score_samples(self._feats(X))


STATE_MODELS = list(M.RECONSTRUCTORS) + ["PCA", "IsolationForest"]


def fit_and_score(name: str, seed: int, trX, vaX, seqs, epochs: int, bs: int, lr: float,
                  dev: str):
    """Train one detector and return (validation scores, per-sequence scores)."""
    if name in M.RECONSTRUCTORS:
        model, tr = train_ae(name, trX, vaX, seed, epochs, bs, lr, device=dev)
        val_sc = ae_scores(model, vaX, device=dev)
        seq_sc = [ae_scores(model, s.X, device=dev) for s in seqs]
        del model
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
    else:
        algo = PCARecon(seed=seed) if name == "PCA" else IForest(seed=seed)
        tr = algo.fit(trX)
        val_sc = classical_scores(algo, vaX)
        seq_sc = [classical_scores(algo, s.X) for s in seqs]
    return val_sc, seq_sc, tr


# --------------------------------------------------------------------------- #
# evaluation (unchanged from the published run)
# --------------------------------------------------------------------------- #
def evaluate_scores(seq_scores, seqs, thresholds, sampling_hz: float) -> dict:
    all_true, all_pred = [], []
    delays, missed, n_segs = [], 0, 0
    thr_list = ([thresholds] * len(seqs) if np.isscalar(thresholds) else list(thresholds))
    for sc, ws, thr in zip(seq_scores, seqs, thr_list):
        keep = ~np.asarray(ws.meta.get("is_calib", np.zeros(len(sc), dtype=bool)))
        pred = (sc > thr).astype(int)[keep]
        true = ws.y.astype(int)[keep]
        all_true.append(true)
        all_pred.append(pred)
        for s, e in MT.find_segments(true):
            n_segs += 1
            hit = np.where(pred[s: e + 1] == 1)[0]
            if len(hit) == 0:
                missed += 1
            else:
                delays.append(int(hit[0]) / sampling_hz)
    yt = np.concatenate(all_true)
    yp = np.concatenate(all_pred)
    res = MT.state_scores(yt, yp, sampling_hz)
    res.update({"n_segments": n_segs, "n_missed": int(missed),
                "missed_rate": float(missed / n_segs) if n_segs else float("nan"),
                "delay_median_s": float(np.median(delays)) if delays else float("nan"),
                "delay_mean_s": float(np.mean(delays)) if delays else float("nan"),
                "delay_p90_s": float(np.percentile(delays, 90)) if delays else float("nan")})
    return res


def slim(res: dict) -> dict:
    """Keep the quantities the paper reports."""
    return {"f1": res["f1"], "pa_f1": res["pa_f1"], "precision": res["precision"],
            "recall": res["recall"], "far": res["false_alarm_rate"],
            "er": 1.0 - res["missed_rate"], "delay_median_s": res["delay_median_s"],
            "n_segments": res["n_segments"]}


def oracle_threshold(seq_sc, seqs) -> float:
    keeps = [~np.asarray(ws.meta.get("is_calib", np.zeros(len(sc), dtype=bool)))
             for sc, ws in zip(seq_sc, seqs)]
    flat_sc = np.concatenate([sc[k] for sc, k in zip(seq_sc, keeps)])
    flat_y = np.concatenate([s.y.astype(int)[k] for s, k in zip(seqs, keeps)])
    thr, _ = MT.best_f1_threshold(flat_sc, flat_y)
    return float(thr)
