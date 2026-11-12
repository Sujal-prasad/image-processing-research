"""Pack a teacher's PNG predictions into the training cache (uint8 array, store order).

    python teachers/pack_cache.py --store data_store/bsds --split trainval --teacher sauge \
        --pred-dir work/pred_sauge [--pred-dir-flip work/pred_sauge_flip] --out teacher_cache/bsds

--pred-dir-flip: predictions on horizontally flipped images; they are flipped back
and averaged with --pred-dir (flip test-time averaging, softens in-sample maps).
Works for any teacher whose outputs are PNG probability maps (8- or 16-bit),
including MEMO (`prediction/` folder of edge_prediction.py) and DiffusionEdge.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pedger_kd.evaluation import load_pred  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--pred-dir-flip", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    d = Path(args.store) / args.split
    names = json.load(open(d / "names.json"))
    transposed = json.load(open(d / "transposed.json"))
    shape = np.load(d / "images.npy", mmap_mode="r").shape[1:3]
    arr = np.zeros((len(names),) + tuple(shape), np.uint8)
    for i, n in enumerate(names):
        prob = load_pred(Path(args.pred_dir) / f"{n}.png")
        if args.pred_dir_flip:
            prob = 0.5 * (prob + load_pred(Path(args.pred_dir_flip) / f"{n}.png")[:, ::-1])
        if transposed[i]:
            prob = prob.T
        if prob.shape != tuple(shape):
            raise ValueError(f"{n}: teacher map {prob.shape} vs store {tuple(shape)}")
        arr[i] = np.clip(np.round(prob * 255), 0, 255).astype(np.uint8)
    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{args.teacher}.npy", arr)
    with open(out / f"{args.teacher}.json", "w") as f:
        json.dump({"names": names, "teacher": args.teacher, "flip_tta": bool(args.pred_dir_flip)}, f)
    print(f"{args.teacher}: {arr.shape} -> {out}; mean prob {arr.mean() / 255:.4f}, "
          f"fraction > 0.5: {(arr > 127).mean():.4f}")


if __name__ == "__main__":
    main()
