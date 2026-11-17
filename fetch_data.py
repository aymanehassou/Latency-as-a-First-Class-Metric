"""Download the three public datasets into the layout rtbench expects."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request

RAW = "https://raw.githubusercontent.com"

WISDM_URL = (RAW + "/curiousily/TensorFlow-on-Android-for-Human-Activity-"
             "Recognition-with-LSTMs/master/data/WISDM_ar_v1.1_raw.txt")
WISDM_MIN_BYTES = 40000000

SKAB_REPO = "https://github.com/waico/SKAB.git"
SKAB_RAW = RAW + "/waico/SKAB/master/data"
SKAB_FILES = ([("anomaly-free", "anomaly-free.csv")]
              + [("valve1", str(i) + ".csv") for i in range(16)]
              + [("valve2", str(i) + ".csv") for i in range(4)]
              + [("other", str(i) + ".csv") for i in range(1, 15)])

SMD_REPO = "https://github.com/NetManAIOps/OmniAnomaly.git"
SMD_RAW = RAW + "/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
SMD_ENTITIES = (["machine-1-" + str(i) for i in range(1, 9)]
                + ["machine-2-" + str(i) for i in range(1, 10)]
                + ["machine-3-" + str(i) for i in range(1, 12)])
SMD_SPLITS = ("train", "test", "test_label")


def log(msg):
    print(msg, flush=True)


def download(url, dest, min_bytes=1):
    if os.path.exists(dest) and os.path.getsize(dest) >= min_bytes:
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
            shutil.copyfileobj(r, fh)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        log("    FAILED " + url + " :: " + str(e))
        return False
    if os.path.getsize(tmp) < min_bytes:
        os.remove(tmp)
        log("    TOO SMALL " + url)
        return False
    os.replace(tmp, dest)
    return True


def clone_subdir(repo, subdir, dest):
    if shutil.which("git") is None:
        return False
    tmp = tempfile.mkdtemp()
    try:
        try:
            subprocess.run(["git", "clone", "--depth", "1", repo, tmp + "\\r"],
                           check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=1800)
        except Exception:
            return False
        src = os.path.join(tmp + "\\r", subdir)
        if not os.path.isdir(src):
            return False
        os.makedirs(dest, exist_ok=True)
        for name in os.listdir(src):
            target = os.path.join(dest, name)
            if os.path.exists(target):
                if os.path.isdir(target):
                    shutil.rmtree(target)
                else:
                    os.remove(target)
            shutil.move(os.path.join(src, name), target)
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fetch_wisdm(root):
    dest = os.path.join(root, "wisdm", "WISDM_ar_v1.1_raw.txt")
    log("WISDM: 1 file, ~47 MB")
    ok = download(WISDM_URL, dest, min_bytes=WISDM_MIN_BYTES)
    log("  " + ("ok" if ok else "FAILED"))
    return ok


def fetch_skab(root, force_raw):
    dest = os.path.join(root, "skab")
    expected = len(SKAB_FILES)
    have = sum(os.path.exists(os.path.join(dest, d, f)) for d, f in SKAB_FILES)
    if have == expected:
        log("SKAB: already present")
        return True
    log("SKAB: " + str(expected) + " files, ~4 MB")
    if not force_raw and clone_subdir(SKAB_REPO, "data", dest):
        have = sum(os.path.exists(os.path.join(dest, d, f)) for d, f in SKAB_FILES)
        if have == expected:
            log("  ok (git clone)")
            return True
        log("  clone incomplete, falling back")
    bad = 0
    for d, f in SKAB_FILES:
        if not download(SKAB_RAW + "/" + d + "/" + f, os.path.join(dest, d, f), 1000):
            bad += 1
    log("  " + str(expected - bad) + "/" + str(expected) + " files")
    return bad == 0


def fetch_smd(root, entities, force_raw):
    dest = os.path.join(root, "smd")
    want = [(s, e) for s in SMD_SPLITS for e in entities]
    have = sum(os.path.exists(os.path.join(dest, s, e + ".txt")) for s, e in want)
    if have == len(want):
        log("SMD: already present")
        return True
    log("SMD: " + str(len(want)) + " files, ~" + str(len(entities) * 17) + " MB")
    if not force_raw and len(entities) == len(SMD_ENTITIES) \
            and clone_subdir(SMD_REPO, "ServerMachineDataset", dest):
        have = sum(os.path.exists(os.path.join(dest, s, e + ".txt")) for s, e in want)
        if have == len(want):
            log("  ok (git clone)")
            return True
        log("  clone incomplete, falling back")
    bad = 0
    for i, e in enumerate(entities, 1):
        for s in SMD_SPLITS:
            if not download(SMD_RAW + "/" + s + "/" + e + ".txt",
                            os.path.join(dest, s, e + ".txt"), 100):
                bad += 1
        if i % 5 == 0 or i == len(entities):
            log("  " + str(i) + "/" + str(len(entities)) + " entities")
    return bad == 0


def verify(root, entities):
    problems = []
    w = os.path.join(root, "wisdm", "WISDM_ar_v1.1_raw.txt")
    if not os.path.exists(w) or os.path.getsize(w) < WISDM_MIN_BYTES:
        problems.append("WISDM raw file missing or truncated")
    for d, f in SKAB_FILES:
        if not os.path.exists(os.path.join(root, "skab", d, f)):
            problems.append("SKAB " + d + "/" + f + " missing")
    for s in SMD_SPLITS:
        for e in entities:
            if not os.path.exists(os.path.join(root, "smd", s, e + ".txt")):
                problems.append("SMD " + s + "/" + e + ".txt missing")
    if problems:
        log("VERIFY FAILED -- " + str(len(problems)) + " problem(s):")
        for p in problems[:10]:
            log("  " + p)
        return False
    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(root) for f in fs)
    log("VERIFY OK -- " + str(round(total / 1e6)) + " MB under " + os.path.abspath(root))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("RTBENCH_DATA", "data"))
    ap.add_argument("--smd-entities", nargs="+", default=SMD_ENTITIES)
    ap.add_argument("--only", nargs="+", choices=["wisdm", "skab", "smd"],
                    default=["wisdm", "skab", "smd"])
    ap.add_argument("--raw", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    ok = True
    if "wisdm" in args.only:
        ok = fetch_wisdm(args.root) and ok
    if "skab" in args.only:
        ok = fetch_skab(args.root, args.raw) and ok
    if "smd" in args.only:
        ok = fetch_smd(args.root, args.smd_entities, args.raw) and ok

    if not ok:
        log("Some downloads failed. Re-run: completed files are skipped.")
        return 1
    if args.only == ["wisdm", "skab", "smd"]:
        return 0 if verify(args.root, args.smd_entities) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
