"""Helpers shared by the teacher runners."""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))


def read_rgb01(path):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).permute(2, 0, 1).float()[None] / 255.0


def save_prob16(path, prob):
    """Save a probability map in [0, 1] as a 16-bit PNG."""
    p = np.clip(np.asarray(prob, dtype=np.float64), 0, 1)
    cv2.imwrite(str(path), (p * 65535 + 0.5).astype(np.uint16))


def crop_for_model(x, mode):
    """mode 'bsds_crop': drop the first row/column (481x321 -> 480x320), as SAUGE/MuGE do on BSDS.
    mode 'pad32': reflect-pad bottom/right to a multiple of 32."""
    H, W = x.shape[-2:]
    if mode == "bsds_crop":
        return x[..., 1:, 1:], ("crop", H, W)
    if mode == "pad32":
        ph, pw = (-H) % 32, (-W) % 32
        return F.pad(x, (0, pw, 0, ph), mode="reflect"), ("pad", H, W)
    if mode == "none":
        return x, ("none", H, W)
    raise ValueError(mode)


def uncrop(prob, info):
    """Back to the original size. For 'crop' the missing first row/column is replicated."""
    kind, H, W = info
    if kind == "crop":
        return F.pad(prob, (1, 0, 1, 0), mode="replicate")
    if kind == "pad":
        return prob[..., :H, :W]
    return prob
