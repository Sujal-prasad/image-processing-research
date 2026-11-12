"""PEdger's loss, computed per image so a micro-batch of B equals B batch-1 steps.

Official loss_function.py (batch size 1):
    w = num_neg / (num_pos + num_neg)        at label == 1
    w = 1.1 * num_pos / (num_pos + num_neg)  at label == 0
    num_pos = sum(label * soft), num_neg = sum((1 - label) * (1 - soft))
    loss = sum over the network's outputs of mean(BCE(pred, soft, weight=w))
The class-balancing counts are taken over the whole tensor, so with B > 1 they
would mix images. Here they are taken per image and the per-image losses averaged.
"""
import torch
import torch.nn.functional as F


def balanced_bce_per_sample(pred, label, soft):
    """pred/label/soft: (B, 1, H, W) float32, pred = probabilities. Returns (B,)."""
    dims = (1, 2, 3)
    npos = (label * soft).sum(dims, keepdim=True)
    nneg = ((1 - label) * (1 - soft)).sum(dims, keepdim=True)
    tot = npos + nneg
    w = torch.where(label == 1, nneg / tot, 1.1 * npos / tot)
    cost = F.binary_cross_entropy(pred, soft, weight=w, reduction="none")
    return cost.mean(dims)


def at_least_fp32(x):
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) or not x.is_floating_point() else x


def pedger_loss(logits, label, soft):
    """logits: list of (B,1,H,W) side/fuse outputs of one PEdger network. Returns scalar."""
    with torch.autocast(device_type=label.device.type, enabled=False):
        label = at_least_fp32(label)
        soft = at_least_fp32(soft)
        per_sample = sum(balanced_bce_per_sample(torch.sigmoid(at_least_fp32(z)), label, soft) for z in logits)
        return per_sample.mean()
