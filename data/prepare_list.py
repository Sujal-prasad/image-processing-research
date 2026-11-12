"""Build an array store split from a list file (NYUDv2, MultiCue, anything else).

List format, one image per line, paths relative to --raw-root:
    <image> <gt_1> [<gt_2> ...]
Each gt can be a BSDS-style .mat (one map per annotator inside), a .mat with a
single 2-D array, or a PNG (> 0 = edge). Several gts per line = several annotators.

Training split:   python data/prepare_list.py --dataset nyud --raw-root NYUD --list train.lst \
                      --split trainval --out data_store/nyud --max-dist 0.011
Evaluation split: add --eval-only (writes eval_lists/<split>.json, no arrays)

Check the downloaded annotation format first (plan: "verify when downloading"):
`python data/visualize_samples.py` shows the annotator fraction maps.
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


def read_list(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if parts:
                rows.append((parts[0], parts[1:]))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--list", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-dist", type=float, default=0.0075)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("H", "W"),
                   help="resize everything to H W (only if sizes differ; not needed for NYUD/MultiCue)")
    args = p.parse_args()
    raw, out = Path(args.raw_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_list(args.list)
    names = [Path(img).stem for img, _ in rows]
    if len(set(names)) != len(names):
        names = [img.replace("/", "__").rsplit(".", 1)[0] for img, _ in rows]

    if args.eval_only:
        (out / "eval_lists").mkdir(exist_ok=True)
        items = []
        for n, (img, gts) in zip(names, rows):
            if len(gts) != 1:
                raise ValueError("eval lists need exactly one gt file per image (multi-annotator .mat is fine)")
            items.append({"name": n, "image": img, "gt": gts[0]})
        with open(out / "eval_lists" / f"{args.split}.json", "w") as f:
            json.dump(items, f, indent=0)
        print(f"wrote {len(items)} eval items")
    else:
        images, fracs, transposed, n_annot = [], {d: [] for d in DILATIONS}, [], []
        for img_rel, gt_rels in rows:
            img = cv2.cvtColor(cv2.imread(str(raw / img_rel)), cv2.COLOR_BGR2RGB)
            gts = [g for rel in gt_rels for g in load_gt(raw / rel)]
            if args.resize:
                H, W = args.resize
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
                gts = [cv2.resize(g.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0 for g in gts]
            t = img.shape[0] > img.shape[1]
            if t:
                img, gts = img.transpose(1, 0, 2), [g.T for g in gts]
            images.append(img)
            for d in DILATIONS:
                fracs[d].append(annotator_fraction(gts, dilate_px=d).astype(np.float16))
            transposed.append(bool(t))
            n_annot.append(len(gts))
        shapes = {im.shape for im in images}
        if len(shapes) != 1:
            raise ValueError(f"mixed image sizes {shapes}; use --resize or crop beforehand")
        d = out / args.split
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "images.npy", np.stack(images))
        for k, v in fracs.items():
            np.save(d / f"frac_d{k}.npy", np.stack(v))
        with open(d / "names.json", "w") as f:
            json.dump(names, f)
        with open(d / "transposed.json", "w") as f:
            json.dump(transposed, f)
        print(f"{args.split}: {len(images)} images {images[0].shape}, annotators "
              f"{min(n_annot)}-{max(n_annot)} (mean {np.mean(n_annot):.2f})")

    meta_path = out / "meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    meta.update({"dataset": args.dataset, "max_dist": args.max_dist, "label_threshold": 0.2,
                 "dilations": list(DILATIONS)})
    meta.setdefault("splits", {})[args.split] = {"n": len(rows), "eval_only": args.eval_only}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
