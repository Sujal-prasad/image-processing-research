"""BSDS-style boundary benchmark (ODS / OIS / AP) through pyEdgeEval.

pyEdgeEval ports the original C++ correspondence code (correspondPixels) and the
edges-toolbox NMS used by RCF's edge_nms.m (r=1, s=5, m=1.01). Protocol:
NMS -> 99 thresholds -> thinning -> matching with maxDist (0.0075 BSDS/MultiCue,
0.011 NYUDv2). It must reproduce a published number before it is trusted (task 1).
Ground truth: BSDS-style .mat (groundTruth{i}.Boundaries, one per annotator),
another .mat holding a single 2-D array, or a PNG (> 0 = edge).
"""
import json
from pathlib import Path

import numpy as np


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


def evaluate(pred_dir, items, max_dist=0.0075, thresholds=99, nproc=4, apply_nms=True, apply_thinning=True):
    """items: list of dicts {name, gt}. Predictions are read from pred_dir/<name>.png."""
    from pyEdgeEval.common.binary_label import calculate_metrics
    samples = [dict(name=it["name"], gt_path=it["gt"], pred_path=str(Path(pred_dir) / f"{it['name']}.png"),
                    thresholds=thresholds, max_dist=max_dist, apply_nms=apply_nms,
                    apply_thinning=apply_thinning) for it in items]
    for s in samples:
        if not Path(s["pred_path"]).exists():
            raise FileNotFoundError(s["pred_path"])
    per_sample, curve, overall = calculate_metrics(eval_single=_eval_one, thresholds=thresholds,
                                                   samples=samples, nproc=nproc)
    scores = {"ODS": float(overall["ODS_f1"]), "OIS": float(overall["OIS_f1"]), "AP": float(overall["AP"]),
              "ODS_threshold": float(overall["ODS_threshold"]), "ODS_recall": float(overall["ODS_recall"]),
              "ODS_precision": float(overall["ODS_precision"]), "n_images": len(samples),
              "max_dist": max_dist, "nms": apply_nms, "thinning": apply_thinning}
    return scores, per_sample, curve


def read_eval_list(store_dir, split, raw_root):
    with open(Path(store_dir) / "eval_lists" / f"{split}.json") as f:
        items = json.load(f)
    raw = Path(raw_root)
    return [{"name": it["name"], "image": str(raw / it["image"]), "gt": str(raw / it["gt"])} for it in items]
