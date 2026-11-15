"""BSDS-style boundary benchmark (ODS / OIS / AP) through pyEdgeEval.

pyEdgeEval ports the original C++ correspondence code (correspondPixels) and the
edges-toolbox NMS used by RCF's edge_nms.m (r=1, s=5, m=1.01). Protocol:
NMS -> 99 thresholds -> thinning -> matching with maxDist (0.0075 BSDS/MultiCue,
0.011 NYUDv2). It must reproduce a published number before it is trusted (task 1).
Ground truth: BSDS-style .mat (groundTruth{i}.Boundaries, one per annotator),
another .mat holding a single 2-D array, or a PNG (> 0 = edge).

Scoring is slow (99 thresholds x every annotator per image: ~4 h for the 200 BSDS test images
on Kaggle's 4 CPUs), so it is resumable: each image's result is appended to a cache file next to
the predictions as soon as it is done, and a later call only scores the images still missing.
The final numbers are computed by pyEdgeEval's own calculate_metrics from those per-image results.
Must stay compatible with Python 3.8 and must not import torch (it also runs in the scorer env).
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Written by tools/install_pyedgeeval.py when pyEdgeEval runs in a separate interpreter
SCORER_MARKER = Path.home() / ".cache" / "pedger_scorer.json"
_UNSET = object()
_SCORER = _UNSET


def _have_pyedgeeval():
    try:
        import pyEdgeEval._lib  # noqa: F401
        return True
    except ImportError:
        return False


def _recorded_scorer():
    try:
        py = json.loads(SCORER_MARKER.read_text())["python"]
        return py if Path(py).exists() else None
    except Exception:
        return None


def ensure_pyedgeeval():
    """Make the benchmark port available. Returns None if pyEdgeEval works in this interpreter, or
    the path of the separate interpreter that has it (fallback set up by tools/install_pyedgeeval.py,
    because `pip install pyEdgeEval` cannot build on Python >= 3.13, which Kaggle uses)."""
    global _SCORER
    if _SCORER is not _UNSET:
        return _SCORER
    if _have_pyedgeeval():
        _SCORER = None
        return None
    py = _recorded_scorer()
    if py is None:
        import importlib
        import subprocess
        import sys
        print("pyEdgeEval is not available: running tools/install_pyedgeeval.py", flush=True)
        subprocess.run([sys.executable, str(ROOT / "tools" / "install_pyedgeeval.py")], check=True)
        importlib.invalidate_caches()
        if _have_pyedgeeval():
            _SCORER = None
            return None
        py = _recorded_scorer()
        if py is None:
            raise RuntimeError("pyEdgeEval installation finished but no working scorer was found")
    _SCORER = py
    return py


def scorer_env():
    """Environment for running a module of this repo under the separate scorer interpreter."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def worker_env():
    """Scorer environment for the worker: one thread per process (the work is split over processes)."""
    env = scorer_env()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[k] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


class ScoringStopped(RuntimeError):
    """Scoring stopped at the deadline. The images scored so far are kept in `cache`; scoring the same
    predictions again (in this or a later session) continues from there."""

    def __init__(self, done, total, cache):
        super().__init__(f"scoring stopped for time after {done}/{total} images (progress kept in {cache})")
        self.done, self.total, self.cache = done, total, Path(cache)


def cache_path(pred_dir, max_dist=0.0075, thresholds=99, apply_nms=True, apply_thinning=True):
    n_thr = thresholds if isinstance(thresholds, int) else len(thresholds)
    return Path(pred_dir) / (f"score_cache_md{max_dist:g}_t{n_thr}_nms{int(bool(apply_nms))}"
                             f"_thin{int(bool(apply_thinning))}.jsonl")


def load_gt(path):
    path = str(path)
    if path.endswith(".mat"):
        import scipy.io as sio
        m = sio.loadmat(path)
        if "groundTruth" in m:
            gt = m["groundTruth"]
            return [np.asarray(gt[0, i]["Boundaries"][0, 0]).astype(bool) for i in range(gt.shape[1])]
        arrays = [v for k, v in m.items() if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2]
        if len(arrays) != 1:
            raise ValueError(f"cannot find a single 2-D ground truth in {path}")
        return [arrays[0] > 0]
    import cv2
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 3:
        im = im[..., 0]
    return [im > 0]


def load_pred(path):
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 3:
        im = im[..., 0]
    scale = 65535.0 if im.dtype == np.uint16 else 255.0
    return im.astype(np.float64) / scale


def _eval_one(kw):
    from pyEdgeEval.common.binary_label.evaluate_boundaries import \
        evaluate_boundaries_threshold_multiple_gts
    from pyEdgeEval.common.utils import check_thresholds
    pred = load_pred(kw["pred_path"])
    gts = load_gt(kw["gt_path"])
    if pred.shape != gts[0].shape:
        raise ValueError(f"{kw['name']}: prediction {pred.shape} vs gt {gts[0].shape}")
    return evaluate_boundaries_threshold_multiple_gts(
        thresholds=check_thresholds(kw["thresholds"]), pred=pred, gts=gts,
        max_dist=kw["max_dist"], apply_thinning=kw["apply_thinning"], apply_nms=kw["apply_nms"],
        nms_kwargs=dict(r=1, s=5, m=1.01, half_prec=False))


def _eval_one_cached(s):
    """Pool task: one image -> (name, prediction checksum, [count_r, sum_r, count_p, sum_p] as lists)."""
    counts = _eval_one(s)
    return s["name"], s["md5"], [np.asarray(c, dtype=np.float64).tolist() for c in counts]


def _file_md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _read_cache(path):
    """{name: (md5, counts)}; unreadable lines (e.g. cut off when a session was killed) are skipped."""
    out = {}
    try:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    out[r["name"]] = (r["md5"], r["counts"])
                except (ValueError, KeyError, TypeError):
                    continue
    except FileNotFoundError:
        pass
    return out


def _open_cache_for_append(path):
    path = Path(path)
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            ends_cleanly = f.read(1) == b"\n"
        if not ends_cleanly:                     # killed mid-write: start a fresh line
            with open(path, "a") as f:
                f.write("\n")
    return open(path, "a")


def score_here(pred_dir, items, max_dist=0.0075, thresholds=99, nproc=1, apply_nms=True, apply_thinning=True,
               cache=None, deadline=None):
    """Score in THIS interpreter (needs pyEdgeEval here). Use evaluate(), which calls this in a worker.
    Returns {"scores", "per_sample", "curve"}, or {"stopped": True, "done", "total"} at the deadline."""
    from multiprocessing import Pool

    from pyEdgeEval.common.binary_label import calculate_metrics
    cache = Path(cache) if cache else cache_path(pred_dir, max_dist, thresholds, apply_nms, apply_thinning)
    samples = [dict(name=it["name"], gt_path=it["gt"], pred_path=str(Path(pred_dir) / f"{it['name']}.png"),
                    thresholds=thresholds, max_dist=max_dist, apply_nms=apply_nms,
                    apply_thinning=apply_thinning) for it in items]
    for s in samples:
        if not Path(s["pred_path"]).exists():
            raise FileNotFoundError(s["pred_path"])
        s["md5"] = _file_md5(s["pred_path"])
    cached = _read_cache(cache)
    results = {s["name"]: cached[s["name"]][1] for s in samples
               if s["name"] in cached and cached[s["name"]][0] == s["md5"]}   # changed predictions are redone
    todo = [s for s in samples if s["name"] not in results]
    total = len(samples)
    print(f"scoring {total} images: {len(results)} already done earlier, {len(todo)} to go, "
          f"{nproc} process(es); a progress line follows every ~5%", flush=True)
    stopped = deadline is not None and time.time() > deadline and bool(todo)
    if todo and not stopped:
        t0, n0, last_print = time.time(), len(results), time.time()
        step = max(1, total // 20)
        pool = Pool(nproc) if nproc > 1 and len(todo) > 1 else None
        try:
            with _open_cache_for_append(cache) as f:
                for name, md5, counts in (pool.imap_unordered(_eval_one_cached, todo) if pool
                                          else map(_eval_one_cached, todo)):
                    results[name] = counts
                    f.write(json.dumps({"name": name, "md5": md5, "counts": counts}) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    n, now = len(results), time.time()
                    if n == total or (n - n0) % step == 0 or now - last_print > 600:
                        per_img = (now - t0) / (n - n0)
                        print(f"  scored {n}/{total} images | {(now - t0) / 60:.0f} min so far | "
                              f"about {per_img * (total - n) / 60:.0f} min left", flush=True)
                        last_print = now
                    if deadline is not None and now > deadline and n < total:
                        stopped = True
                        break
        finally:
            if pool is not None:
                pool.terminate()
                pool.join()
    if stopped:
        print(f"  stopping for time: {len(results)}/{total} images done (kept in {cache.name})", flush=True)
        return {"stopped": True, "done": len(results), "total": total}
    lookup = {name: tuple(np.asarray(c, dtype=np.float64) for c in counts) for name, counts in results.items()}
    # pyEdgeEval's own aggregation (ODS interpolation, OIS, AP) on the per-image results
    per_sample, curve, overall = calculate_metrics(eval_single=lambda s: lookup[s["name"]], thresholds=thresholds,
                                                   samples=samples, nproc=1)
    for s in per_sample:
        s.pop("md5", None)
    scores = {"ODS": float(overall["ODS_f1"]), "OIS": float(overall["OIS_f1"]), "AP": float(overall["AP"]),
              "ODS_threshold": float(overall["ODS_threshold"]), "ODS_recall": float(overall["ODS_recall"]),
              "ODS_precision": float(overall["ODS_precision"]), "n_images": len(samples),
              "max_dist": max_dist, "nms": apply_nms, "thinning": apply_thinning}
    return {"scores": scores, "per_sample": per_sample, "curve": curve}


def evaluate(pred_dir, items, max_dist=0.0075, thresholds=99, nproc=None, apply_nms=True, apply_thinning=True,
             deadline=None, on_tick=None, tick_minutes=10.0):
    """Score pred_dir/<name>.png against each item's ground truth (items: dicts {name, gt}).

    Runs in a fresh worker process (with pyEdgeEval from this Python or from the separate scorer
    environment), so its process pool never forks a process that already ran PyTorch/OpenCV threads.
    Resumable: per-image results are kept in cache_path(pred_dir, ...). Raises ScoringStopped when
    `deadline` (epoch seconds) is reached first. on_tick(cache_file) is called every `tick_minutes`
    while the worker runs (e.g. to upload the progress). Returns (scores, per_sample, curve)."""
    python = ensure_pyedgeeval() or sys.executable
    nproc = int(nproc or os.cpu_count() or 1)
    cache = cache_path(pred_dir, max_dist, thresholds, apply_nms, apply_thinning)
    spec = dict(pred_dir=str(pred_dir), items=[{"name": it["name"], "gt": str(it["gt"])} for it in items],
                max_dist=max_dist, thresholds=thresholds, nproc=nproc, apply_nms=apply_nms,
                apply_thinning=apply_thinning, cache=str(cache), deadline=deadline)
    with tempfile.TemporaryDirectory() as d:
        sp, out = Path(d) / "spec.json", Path(d) / "out.json"
        sp.write_text(json.dumps(spec))
        proc = subprocess.Popen([python, "-m", "pedger_kd.score_worker", str(sp), str(out)], cwd=str(ROOT),
                                env=worker_env())
        last_tick = time.time()
        try:
            while True:
                try:
                    rc = proc.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    if on_tick is not None and time.time() - last_tick >= tick_minutes * 60:
                        last_tick = time.time()
                        try:
                            on_tick(cache)
                        except Exception as e:      # never lose the scoring because an upload failed
                            print(f"  (saving progress failed: {e}; will try again)", flush=True)
        finally:
            if proc.poll() is None:
                proc.terminate()
        if rc != 0:
            raise RuntimeError(f"the scoring worker failed (exit code {rc}); its error is printed above")
        res = json.loads(out.read_text())
    if res.get("stopped"):
        raise ScoringStopped(res["done"], res["total"], cache)
    return res["scores"], res["per_sample"], res["curve"]


def read_eval_list(store_dir, split, raw_root):
    with open(Path(store_dir) / "eval_lists" / f"{split}.json") as f:
        items = json.load(f)
    raw = Path(raw_root)
    return [{"name": it["name"], "image": str(raw / it["image"]), "gt": str(raw / it["gt"])} for it in items]
