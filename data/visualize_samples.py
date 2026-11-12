"""Visual checks.

Task 2/3  (default): image | annotator fraction a | label Y | trust alpha | teacher T
Task 6    (--augmented): the same AFTER joint augmentation, plus an overlay of the
          teacher map (red) and the label (green) on the image. If augmentation is
          aligned, red and green trace the same contours.

    python data/visualize_samples.py --store data_store/bsds --split train --n 10 \
        --teacher-root teacher_cache/bsds --teacher ensemble_mean --augmented --out checks/aug.png
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pedger_kd.agreement import human_trust  # noqa: E402
from pedger_kd.augment import IMAGENET_MEAN, IMAGENET_STD, JointAugment  # noqa: E402
from pedger_kd.data import ArrayStore  # noqa: E402


def gray(x):
    x = np.clip(np.asarray(x, dtype=np.float32), 0, 1)
    return cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--teacher-root", default=None)
    p.add_argument("--teacher", default=None)
    p.add_argument("--alpha-min", type=float, default=0.1)
    p.add_argument("--augmented", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    st = ArrayStore(args.store, args.split, args.teacher_root, args.teacher, device="cpu")
    idx = torch.randperm(len(st))[: args.n]
    rows = []
    aug = JointAugment({}, train=args.augmented)
    for i in idx.tolist():
        img, frac, frac_a, T = st.get([i])
        b = aug(img, frac, frac_a, T)
        im = b["img"][0] * torch.tensor(IMAGENET_STD)[:, None, None] + torch.tensor(IMAGENET_MEAN)[:, None, None]
        im = cv2.cvtColor((im.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        a = b["a"][0, 0].numpy()
        tiles = [im, gray(a), gray(b["Y"][0, 0].numpy()), gray(human_trust(b["a_alpha"][0, 0].numpy(), args.alpha_min))]
        if b["T"] is not None:
            t = b["T"][0, 0].numpy()
            tiles.append(gray(t))
            ov = (im * 0.5).astype(np.uint8)
            ov[..., 2] = np.maximum(ov[..., 2], (t * 255).astype(np.uint8))
            ov[..., 1] = np.maximum(ov[..., 1], (b["Y"][0, 0].numpy() * 255).astype(np.uint8))
            tiles.append(ov)
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                 for t in tiles]
        rows.append(np.concatenate(tiles, 1))
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 4, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, np.concatenate(rows, 0))
    cols = "image | a | Y | alpha" + (" | T | overlay (red=T, green=Y)" if st.teacher is not None else "")
    print(f"wrote {args.out}: {cols}")


if __name__ == "__main__":
    main()
