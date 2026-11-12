"""Teacher ensemble T (plan 8.4): mean (default) or per-pixel confidence-weighted.

Training cache:  python teachers/ensemble.py --teacher-root teacher_cache/bsds --split trainval \
                     --teachers sauge muge memo --mode mean --name ensemble_mean
PNG folders (teacher gate on a validation split):
                 python teachers/ensemble.py --png-dirs work/val_sauge work/val_muge --out-dir work/val_ensemble
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pedger_kd.evaluation import load_pred  # noqa: E402
from pedger_kd.teachers import teacher_ensemble  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="mean", choices=["mean", "confidence"])
    p.add_argument("--teacher-root")
    p.add_argument("--split")
    p.add_argument("--teachers", nargs="+")
    p.add_argument("--name", default=None)
    p.add_argument("--png-dirs", nargs="+")
    p.add_argument("--out-dir")
    args = p.parse_args()
    weighted = args.mode == "confidence"
    if args.png_dirs:
        from common import save_prob16
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = sorted(q.stem for q in Path(args.png_dirs[0]).glob("*.png"))
        for n in names:
            maps = [load_pred(Path(d) / f"{n}.png") for d in args.png_dirs]
            save_prob16(out / f"{n}.png", teacher_ensemble(maps, weighted=weighted))
        print(f"wrote {len(names)} ensemble maps to {out}")
        return
    d = Path(args.teacher_root) / args.split
    names, maps = None, []
    for t in args.teachers:
        meta = json.load(open(d / f"{t}.json"))
        if names is not None and meta["names"] != names:
            raise ValueError(f"{t} is not aligned with {args.teachers[0]}")
        names = meta["names"]
        maps.append(np.load(d / f"{t}.npy").astype(np.float32) / 255.0)
    T = np.stack([teacher_ensemble([m[i] for m in maps], weighted=weighted) for i in range(len(names))])
    name = args.name or f"ensemble_{args.mode}"
    np.save(d / f"{name}.npy", np.clip(np.round(T * 255), 0, 255).astype(np.uint8))
    with open(d / f"{name}.json", "w") as f:
        json.dump({"names": names, "teacher": name, "members": args.teachers, "mode": args.mode}, f)
    print(f"{name} = {args.mode}({', '.join(args.teachers)}) -> {d / (name + '.npy')}")


if __name__ == "__main__":
    main()
