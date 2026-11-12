"""Per-image normalisation so that a micro-batch of B images trains exactly like
PEdger's batch size of 1.

PEdger's BatchNorm2d layers use track_running_stats=False, so they always
normalise with the statistics of the current batch. PEdger trains and tests one
image at a time, which makes every BN layer compute per-image statistics, i.e.
behave like InstanceNorm with BN's affine parameters.

If we simply fed B images at once, BN would pool statistics across the B images
and training would no longer match test-time behaviour (the "BatchNorm check").
`PerSampleBatchNorm2d` keeps the same parameters and state_dict keys but
normalises each image on its own, so a batch of B is numerically equivalent to B
separate batch-1 forward passes. Checkpoints stay loadable by the official code.
"""
import torch.nn as nn
import torch.nn.functional as F


class PerSampleBatchNorm2d(nn.BatchNorm2d):
    def forward(self, x):
        return F.instance_norm(
            x, running_mean=None, running_var=None, weight=self.weight, bias=self.bias,
            use_input_stats=True, momentum=0.0, eps=self.eps)


def convert_to_per_sample(module):
    """In-place: switch every stats-free BatchNorm2d to per-sample statistics."""
    n = 0
    for m in module.modules():
        if type(m) is nn.BatchNorm2d:
            if m.track_running_stats:
                raise ValueError("Expected PEdger BN layers with track_running_stats=False")
            m.__class__ = PerSampleBatchNorm2d
            n += 1
    return n


def convert_to_stock(module):
    """Undo convert_to_per_sample (used by the BatchNorm check)."""
    n = 0
    for m in module.modules():
        if type(m) is PerSampleBatchNorm2d:
            m.__class__ = nn.BatchNorm2d
            n += 1
    return n


def set_norm_mode(module, mode):
    if mode == "per_sample":
        return convert_to_per_sample(module)
    if mode == "stock":
        return convert_to_stock(module)
    raise ValueError(f"unknown norm mode {mode!r}")
