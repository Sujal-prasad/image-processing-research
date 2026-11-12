"""Variant B: a separate, weighted distillation term on top of the unchanged PEdger loss.

L = L_PEdger + lambda_kd * sum_p w(p) BCE(sigmoid(z_S/tau), sigmoid(z_T/tau)) / sum_p w(p)
w(p) = 1 - alpha(p) + w_min   (agreement)   or   1   (uniform)
Optional logit standardisation (per-image z-score, adapted from Sun et al. CVPR 2024).
"""
import torch
import torch.nn.functional as F

from .agreement import human_trust


def standardize(z, eps=1e-6):
    return (z - z.mean((2, 3), keepdim=True)) / (z.std((2, 3), keepdim=True) + eps)


def kd_loss(z_s, p_t, w, tau=2.0, use_ls=True):
    z_s = z_s.float()
    z_t = torch.logit(p_t.float().clamp(1e-4, 1 - 1e-4))
    if use_ls:
        z_s, z_t = standardize(z_s), standardize(z_t)
    loss = F.binary_cross_entropy_with_logits(z_s / tau, torch.sigmoid(z_t / tau), reduction="none")
    return (w * loss).sum() / w.sum().clamp_min(1.0)


def kd_weight(kd_cfg, method_cfg, a_alpha):
    mode = kd_cfg.get("weight", "uniform")
    if mode == "uniform":
        return torch.ones_like(a_alpha)
    if mode == "agreement":
        alpha = human_trust(a_alpha, method_cfg.get("alpha_min", 0.1))
        return 1 - alpha + kd_cfg.get("w_min", 0.1)
    raise ValueError(f"unknown kd weight {mode!r}")
