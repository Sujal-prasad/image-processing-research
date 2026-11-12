"""Agreement-bin analysis (step 13, evidence for H2).

For each test image, predictions are thresholded at the model's ODS threshold
after the same NMS + thinning as the benchmark, then matched against every
annotator with the benchmark's correspondence code.

a(p) = fraction of annotators marking pixel p (from the original per-annotator .mat).
Recall per bin   : ground-truth pixels of each annotator, binned by a(p) at that pixel;
                   recall = matched / total.
Precision per bin: predicted pixels, binned by a_near(p) = max a within the match
                   radius (maxDist * image diagonal); 'none' = no annotator nearby.
Bins: low (0, 1/3], medium (1/3, 2/3], high (2/3, 1].
If H2 holds, our gain over PEdger concentrates in the low-agreement bin.

    python eval/agreement_bins.py --store data_store/bsds --split test --raw-root BSR/BSDS500/data \
        --run PEdger=runs/baseline_s0/scores_test_ns_m.json --run Ours=runs/A_agree_s0/scores_test_ns_m.json \
        --out results/agreement_bins.json
"""
import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pedger_kd.data import read_meta  # noqa: E402
from pedger_kd.evaluation import load_gt, load_pred, read_eval_list  # noqa: E402

BINS = {"low": (0.0, 1 / 3), "medium": (1 / 3, 2 / 3), "high": (2 / 3, 1.0)}


def bin_masks(a, include_none=False):
    masks = {k: (a > lo) & (a <= hi + 1e-9) for k, (lo, hi) in BINS.items()}
    if include_none:
        masks["none"] = a <= 0
    return masks


def analyse_image(kw):
    from pyEdgeEval._lib import correspond_pixels
    from pyEdgeEval.preprocess import binary_thin, fast_nms
    pred = fast_nms(img=load_pred(kw["pred"]), r=1, s=5, m=1.01, half_prec=False)
    pb = binary_thin(pred >= kw["thr"])
    gts = load_gt(kw["gt"])
    a = np.mean(np.stack(gts).astype(np.float32), 0)
    radius = max(1, int(math.ceil(kw["max_dist"] * math.hypot(*a.shape))))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    a_near = cv2.dilate(a, disk)
    res = {k: [0, 0, 0, 0] for k in list(BINS) + ["none"]}  # rec_hit, rec_tot, prec_hit, prec_tot
    matched_pred = np.zeros_like(pb, bool)
    for g in gts:
        m1, m2, _, _ = correspond_pixels(pb, g, max_dist=kw["max_dist"])
        matched_pred |= m1 > 0
        for k, msk in bin_masks(a).items():
            sel = g & msk
            res[k][0] += int(((m2 > 0) & sel).sum())
            res[k][1] += int(sel.sum())
    for k, msk in bin_masks(a_near, include_none=True).items():
        sel = pb & msk
        res[k][2] += int((matched_pred & sel).sum())
        res[k][3] += int(sel.sum())
    return res


def run_bins(pred_dir, thr, items, max_dist, nproc):
    jobs = [{"pred": str(Path(pred_dir) / f"{it['name']}.png"), "gt": it["gt"], "thr": thr,
             "max_dist": max_dist} for it in items]
    with Pool(nproc) as pool:
        outs = pool.map(analyse_image, jobs)
    tot = {k: [0, 0, 0, 0] for k in list(BINS) + ["none"]}
    for o in outs:
        for k, v in o.items():
            tot[k] = [x + y for x, y in zip(tot[k], v)]
    table = {}
    for k, (rh, rt, ph, pt) in tot.items():
        table[k] = {"recall": rh / rt if rt else None, "gt_pixels": rt,
                    "precision": ph / pt if pt else None, "pred_pixels": pt}
    return table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--raw-root", required=True)
    p.add_argument("--run", action="append", required=True,
                   help="label=scores json (gives pred_dir and ODS threshold)")
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    meta = read_meta(args.store)
    items = read_eval_list(args.store, args.split, args.raw_root)
    result = {}
    for spec in args.run:
        label, path = spec.split("=", 1)
        sc = json.load(open(path))
        result[label] = run_bins(sc["pred_dir"], sc["ODS_threshold"], items, meta["max_dist"], args.nproc)
        result[label]["_ODS"] = sc["ODS"]
    print(f"{'bin':8s}" + "".join(f"{lab + ' R':>14s}{lab + ' P':>14s}" for lab in result))
    for k in list(BINS) + ["none"]:
        row = f"{k:8s}"
        for lab in result:
            r, pr = result[lab][k]["recall"], result[lab][k]["precision"]
            row += f"{(r if r is not None else float('nan')):14.4f}{(pr if pr is not None else float('nan')):14.4f}"
        print(row)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
