"""Array store produced by data/prepare_*.py, plus the cached teacher maps.

Store layout (one folder per dataset):
    meta.json
    <split>/images.npy      (N, H, W, 3) uint8, all in one canonical orientation
    <split>/frac_d0.npy     (N, H, W) float16   fraction of annotators marking each pixel
    <split>/frac_d1.npy     same, each annotator's map dilated by 1 px before averaging
    <split>/frac_d2.npy     same, 2 px
    <split>/names.json      image ids in array order
Teacher cache (teachers/pack_cache.py):
    <teacher_root>/<split>/<teacher>.npy   (N, H, W) uint8 probability * 255, same order
    <teacher_root>/<split>/<teacher>.json  names, for an alignment check
Everything is small enough (BSDS: < 0.5 GB) to sit on the GPU, so there is no
DataLoader and no CPU decoding during training.
"""
import json
from pathlib import Path

import numpy as np
import torch


def read_meta(store_dir):
    with open(Path(store_dir) / "meta.json") as f:
        return json.load(f)


def split_names(store_dir, split):
    with open(Path(store_dir) / split / "names.json") as f:
        return json.load(f)


class ArrayStore:
    def __init__(self, store_dir, split, teacher_dir=None, teacher=None, dilation=0,
                 device="cpu", on_gpu=True):
        d = Path(store_dir) / split
        self.names = split_names(store_dir, split)
        self.device = torch.device(device)
        place = self.device if on_gpu else torch.device("cpu")

        def load(path, dtype=None):
            t = torch.from_numpy(np.array(np.load(path, mmap_mode="r")))
            if dtype is not None:
                t = t.to(dtype)
            return t.to(place)

        self.images = load(d / "images.npy")
        self.frac = load(d / "frac_d0.npy", torch.float16)
        self.frac_alpha = self.frac if dilation == 0 else load(d / f"frac_d{dilation}.npy", torch.float16)
        self.teacher = None
        if teacher:
            tpath = Path(teacher_dir) / split / f"{teacher}.npy"
            if not tpath.exists():
                raise FileNotFoundError(f"teacher cache missing: {tpath}")
            jpath = tpath.with_suffix(".json")
            if jpath.exists():
                with open(jpath) as f:
                    tnames = json.load(f)["names"]
                if tnames != self.names:
                    raise ValueError(f"teacher cache {tpath} is not aligned with store split {split}")
            self.teacher = load(tpath)
            if tuple(self.teacher.shape) != tuple(self.images.shape[:3]):
                raise ValueError(f"teacher shape {tuple(self.teacher.shape)} != images {tuple(self.images.shape[:3])}")

    def __len__(self):
        return len(self.names)

    def get(self, idx):
        idx = torch.as_tensor(idx, dtype=torch.long)
        def take(t):
            if t is None:
                return None
            out = t[idx.to(t.device)]
            return out.to(self.device, non_blocking=True)
        return take(self.images), take(self.frac), take(self.frac_alpha), take(self.teacher)
