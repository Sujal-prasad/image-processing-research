"""Build the BSDS500 array store (coding tasks 2-3).

Input: the original Berkeley release (BSR_bsds500.tgz -> BSR/BSDS500/data):
    images/{train,val,test}/*.jpg, groundTruth/{train,val,test}/*.mat
Each .mat holds one boundary map per annotator, so the annotator fraction a(p)
is computed exactly here (no reliance on pre-averaged PNGs).

Output store (see pedger_kd/data.py): train (200), val (100), trainval (300)
arrays in landscape orientation (portrait images are transposed; a transpose is
a rotation+flip, i.e. inside the augmentation group), plus eval lists for
val/test pointing at the original files.

    python data/prepare_bsds.py --bsds-root BSR/BSDS500/data --out data_store/bsds
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pedger_kd.agreement import annotator_fraction  # noqa: E402
from pedger_kd.evaluation import load_gt  # noqa: E402

DILATIONS = (0, 1, 2)


def build_split(root, split):
    names = sorted(p.stem for p in (root / "images" / split).glob("*.jpg"))
    if not names:
        raise FileNotFoundError(f"no images in {root / 'images' / split}")
    images, fracs, transposed, n_annot = [], {d: [] for d in DILATIONS}, [], []
    for n in names:
        img = cv2.cvtColor(cv2.imread(str(root / "images" / split / f"{n}.jpg")), cv2.COLOR_BGR2RGB)
        gts = load_gt(root / "groundTruth" / split / f"{n}.mat")
        assert gts[0].shape == img.shape[:2], n
        t = img.shape[0] > img.shape[1]
        if t:
            img = img.transpose(1, 0, 2)
            gts = [g.T for g in gts]
        images.append(img)
        for d in DILATIONS:
            fracs[d].append(annotator_fraction(gts, dilate_px=d).astype(np.float16))
        transposed.append(bool(t))
        n_annot.append(len(gts))
    shapes = {im.shape for im in images}
    if len(shapes) != 1:
        raise ValueError(f"{split}: mixed sizes after canonicalisation: {shapes}")
    return names, np.stack(images), {d: np.stack(v) for d, v in fracs.items()}, transposed, n_annot


def save_split(out, split, names, images, fracs, transposed):
    d = out / split
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "images.npy", images)
    for k, v in fracs.items():
        np.save(d / f"frac_d{k}.npy", v)
    with open(d / "names.json", "w") as f:
        json.dump(names, f)
    with open(d / "transposed.json", "w") as f:
        json.dump(transposed, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bsds-root", required=True, help=".../BSR/BSDS500/data")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root, out = Path(args.bsds_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    built, stats = {}, {}
    for split in ("train", "val"):
        names, images, fracs, transposed, n_annot = build_split(root, split)
        save_split(out, split, names, images, fracs, transposed)
        built[split] = (names, images, fracs, transposed)
        vals = np.unique(fracs[0][fracs[0] > 0].astype(np.float32))
        stats[split] = {"n": len(names), "shape": list(images.shape[1:3]),
                        "annotators_min": int(min(n_annot)), "annotators_max": int(max(n_annot)),
                        "annotators_mean": float(np.mean(n_annot)),
                        "edge_fraction_values_example": [round(float(v), 3) for v in vals[:12]]}
        print(split, stats[split])
    tv = [built["train"], built["val"]]
    save_split(out, "trainval", tv[0][0] + tv[1][0], np.concatenate([tv[0][1], tv[1][1]]),
               {d: np.concatenate([tv[0][2][d], tv[1][2][d]]) for d in DILATIONS}, tv[0][3] + tv[1][3])
    stats["trainval"] = {"n": len(tv[0][0]) + len(tv[1][0])}

    (out / "eval_lists").mkdir(exist_ok=True)
    for split in ("val", "test"):
        names = sorted(p.stem for p in (root / "images" / split).glob("*.jpg"))
        items = [{"name": n, "image": f"images/{split}/{n}.jpg", "gt": f"groundTruth/{split}/{n}.mat"}
                 for n in names]
        with open(out / "eval_lists" / f"{split}.json", "w") as f:
            json.dump(items, f, indent=0)
        stats[f"eval_{split}"] = len(items)

    meta = {"dataset": "bsds", "max_dist": 0.0075, "label_threshold": 0.2, "dilations": list(DILATIONS),
            "orientation": "landscape (portrait images transposed, see <split>/transposed.json)",
            "raw_root_hint": "pass --raw-root pointing at BSR/BSDS500/data for evaluation", "splits": stats}
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Store written to {out}")


if __name__ == "__main__":
    main()
