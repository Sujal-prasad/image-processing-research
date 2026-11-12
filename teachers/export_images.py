"""Write the images teachers must run on, as lossless PNGs named <name>.png.

Training split (from the store, restored to original orientation):
    python teachers/export_images.py --store data_store/bsds --split trainval --out work/img_trainval
    (+ --flip writes horizontally flipped copies to <out>_flip for flip test-time averaging)
Evaluation split (for the teacher gate), from the original files:
    python teachers/export_images.py --store data_store/nyud --eval-split val --raw-root NYUD --out work/img_val
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pedger_kd.evaluation import read_eval_list  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", required=True)
    p.add_argument("--split", default=None)
    p.add_argument("--eval-split", default=None)
    p.add_argument("--raw-root", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--flip", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    outf = Path(str(out) + "_flip")
    if args.flip:
        outf.mkdir(parents=True, exist_ok=True)

    def write(name, rgb):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out / f"{name}.png"), bgr)
        if args.flip:
            cv2.imwrite(str(outf / f"{name}.png"), bgr[:, ::-1])

    if args.split:
        d = Path(args.store) / args.split
        images = np.load(d / "images.npy", mmap_mode="r")
        names = json.load(open(d / "names.json"))
        transposed = json.load(open(d / "transposed.json"))
        for i, n in enumerate(names):
            img = np.asarray(images[i])
            write(n, img.transpose(1, 0, 2) if transposed[i] else img)
        print(f"exported {len(names)} training images to {out}")
    elif args.eval_split:
        items = read_eval_list(args.store, args.eval_split, args.raw_root)
        for it in items:
            write(it["name"], cv2.cvtColor(cv2.imread(it["image"]), cv2.COLOR_BGR2RGB))
        print(f"exported {len(items)} eval images to {out}")
    else:
        raise SystemExit("give --split or --eval-split")


if __name__ == "__main__":
    main()
