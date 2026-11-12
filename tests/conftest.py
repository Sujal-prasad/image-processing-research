import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pedger_kd.config import load_config  # noqa: E402

HAS_PEDGER = (ROOT / "third_party" / "PEdger" / "models.py").exists()
needs_pedger = pytest.mark.skipif(not HAS_PEDGER, reason="run: python tools/fetch_upstream.py")


def base_cfg(**over):
    cfg = load_config(ROOT / "configs" / "base.yaml")
    cfg["train"].update({"amp": False, "channels_last": False})
    for k, v in over.items():
        cfg[k] = {**cfg.get(k, {}), **v} if isinstance(v, dict) else v
    return cfg


def make_batch(B=2, H=48, W=64, seed=0, teacher=True):
    g = torch.Generator().manual_seed(seed)
    a = (torch.rand(B, 1, H, W, generator=g) < 0.08).float() * torch.randint(1, 6, (B, 1, H, W), generator=g) / 5
    return {"img": torch.randn(B, 3, H, W, generator=g), "Y": (a >= 0.2).float(), "a": a, "a_alpha": a,
            "T": torch.rand(B, 1, H, W, generator=g) if teacher else None}


@pytest.fixture
def tiny_store(tmp_path):
    """A 4-image synthetic store + teacher cache with the real on-disk layout."""
    rng = np.random.default_rng(0)
    N, H, W = 4, 64, 96
    store = tmp_path / "data" / "toy"
    d = store / "train"
    d.mkdir(parents=True)
    imgs = rng.integers(0, 255, (N, H, W, 3), dtype=np.uint8)
    frac = np.zeros((N, H, W), np.float16)
    frac[:, 20, :] = 0.6
    frac[:, :, 40] = 1.0
    frac[:, 45, 10:80] = 0.2
    np.save(d / "images.npy", imgs)
    for k in (0, 1, 2):
        np.save(d / f"frac_d{k}.npy", frac)
    names = [f"img{i}" for i in range(N)]
    json.dump(names, open(d / "names.json", "w"))
    json.dump([False] * N, open(d / "transposed.json", "w"))
    json.dump({"dataset": "toy", "max_dist": 0.0075}, open(store / "meta.json", "w"))
    tdir = tmp_path / "teachers" / "toy" / "train"
    tdir.mkdir(parents=True)
    np.save(tdir / "ensemble_mean.npy", (frac.astype(np.float32) * 200).astype(np.uint8))
    json.dump({"names": names}, open(tdir / "ensemble_mean.json", "w"))
    return tmp_path
