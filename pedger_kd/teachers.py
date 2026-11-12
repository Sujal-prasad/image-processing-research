"""Teacher ensemble (plan section 8.4). Works on numpy arrays or torch tensors."""
import numpy as np


def teacher_ensemble(maps, weighted=False, eps=1e-6):
    """maps: list of same-shape probability maps in [0, 1]."""
    if isinstance(maps[0], np.ndarray):
        T = np.stack([m.astype(np.float32) for m in maps], 0)
        if not weighted:
            return T.mean(0)
        c = np.abs(2 * T - 1) + eps
        return (c * T).sum(0) / c.sum(0)
    import torch
    T = torch.stack([m.float() for m in maps], 0)
    if not weighted:
        return T.mean(0)
    c = (2 * T - 1).abs() + eps
    return (c * T).sum(0) / c.sum(0)
