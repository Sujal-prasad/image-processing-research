"""Human-agreement maps (plan section 8.2).

a(p)     fraction of annotators who marked pixel p as an edge, in [0, 1]
agree(p) = |2 a(p) - 1|        1 = everyone agrees (edge or no edge), 0 = evenly split
alpha(p) = a_min + (1 - a_min) * agree(p)     trust in the human label

All functions work on numpy arrays and torch tensors alike.
"""
import numpy as np


def annotator_fraction(boundaries, dilate_px=0):
    """boundaries: list of (H, W) binary maps, one per annotator.

    dilate_px > 0 dilates each annotator's map with a disk before averaging,
    because hand-drawn lines are often offset by a pixel or two (ablation).
    """
    maps = [np.asarray(b) > 0 for b in boundaries]
    if not maps:
        raise ValueError("no annotator maps")
    if dilate_px > 0:
        import cv2
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        maps = [cv2.dilate(m.astype(np.uint8), k) > 0 for m in maps]
    return np.mean(np.stack(maps, 0).astype(np.float32), axis=0)


def agreement(a):
    return abs(2 * a - 1)


def human_trust(a, a_min=0.1):
    return a_min + (1 - a_min) * agreement(a)


def teacher_conf_trust(T, c=0.8):
    """Single-annotation datasets (NYUDv2): trust the label less where the teacher is confident."""
    return 1 - c * abs(2 * T - 1)


def binarize_label(a, threshold=0.2):
    """PEdger's label: edge_gt >= 0.2 on the averaged ground truth (datasets.py)."""
    return (a >= threshold).astype(np.float32) if isinstance(a, np.ndarray) else (a >= threshold).float()
