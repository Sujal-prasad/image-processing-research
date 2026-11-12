"""Inference, matching the official test.py.

Modes (which network(s) produce the edge map):
  ns_m    non-recurrent momentum network  (G_teacher_noshare, 0.73M params) <- deployed model
  rec_m   recurrent momentum network      (G_teacher, 0.36M)
  fuse_m  both momentum networks fused by confidence (2 networks, not 0.73M)
  ns / rec / fuse   the same with the non-momentum (G, G_noshare) weights
Output transform:
  pedger   exp(z-.5)/(exp(z-.5)+exp(.5-z)) = sigmoid(2z - 1)   (official test.py)
  sigmoid  sigmoid(z)
Multi-scale: scales (0.5, 1, 1.5); logits are resized back to full size, (fused,)
transformed and averaged, exactly as in test.py.
"""
import torch
import torch.nn.functional as F

from . import upstream
from .norm import set_norm_mode
from .train_kd import NET_KEYS

MODES = {"ns_m": ["ns_m"], "rec_m": ["rec_m"], "fuse_m": ["rec_m", "ns_m"],
         "ns": ["ns"], "rec": ["rec"], "fuse": ["rec", "ns"]}


def load_nets(ckpt_path, device="cpu", norm="per_sample", mode="ns_m"):
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    nets = {}
    for k in MODES[mode]:
        net = upstream.build_recurrent() if k.startswith("rec") else upstream.build_noshare()
        net.load_state_dict(st[NET_KEYS[k]])
        set_norm_mode(net, norm)
        nets[k] = net.to(device).eval()
    return nets


def output_transform(z, kind="pedger"):
    if kind == "pedger":
        return torch.sigmoid(2 * z - 1)
    if kind == "sigmoid":
        return torch.sigmoid(z)
    raise ValueError(kind)


@torch.no_grad()
def predict(nets, x, mode="ns_m", transform="pedger", scales=(1.0,)):
    """x: (1,3,H,W) normalised image. Returns (H, W) probabilities as a tensor."""
    H, W = x.shape[-2:]
    keys = MODES[mode]
    probs = []
    for s in scales:
        xs = x if s == 1.0 else F.interpolate(x, size=(int(round(H * s)), int(round(W * s))), mode="bilinear")
        zs = []
        for k in keys:
            z = nets[k](xs)[-1].float()
            if s != 1.0:
                z = F.interpolate(z, size=(H, W), mode="bilinear")
            zs.append(z)
        if len(zs) == 1:
            z = zs[0]
        else:
            u_r = torch.abs(torch.sigmoid(zs[0]) - 0.5)
            u_n = torch.abs(torch.sigmoid(zs[1]) - 0.5)
            w = u_r / (u_r + u_n).clamp_min(1e-12)
            z = zs[0] * w + zs[1] * (1 - w)
        probs.append(output_transform(z, transform))
    return torch.stack(probs, 0).mean(0)[0, 0]
