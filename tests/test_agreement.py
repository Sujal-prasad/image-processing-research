import numpy as np
import torch

from pedger_kd.agreement import (agreement, annotator_fraction, binarize_label, human_trust,
                                 teacher_conf_trust)
from pedger_kd.kd_loss import kd_weight


def test_trust_extremes():
    a = np.array([0.0, 1.0, 0.5, 0.2, 0.8])
    al = human_trust(a, a_min=0.1)
    assert np.allclose(al[:2], 1.0)          # everyone agrees -> trust the label fully
    assert np.isclose(al[2], 0.1)            # evenly split -> a_min
    assert np.isclose(al[3], al[4])          # symmetric
    assert np.allclose(agreement(a), [1, 1, 0, 0.6, 0.6])


def test_fraction_and_dilation():
    m1 = np.zeros((9, 9), bool); m1[4, :] = True
    m2 = np.zeros((9, 9), bool); m2[5, :] = True   # same line, one pixel lower
    a0 = annotator_fraction([m1, m2])
    assert a0[4, 3] == 0.5 and a0[5, 3] == 0.5    # offset lines look like disagreement
    a1 = annotator_fraction([m1, m2], dilate_px=1)
    assert a1[4, 3] == 1.0 and a1[5, 3] == 1.0    # dilation recovers the agreement


def test_label_threshold_matches_pedger():
    a = np.array([0.0, 0.19, 0.2, 0.6])
    assert binarize_label(a).tolist() == [0, 0, 1, 1]


def test_teacher_conf_and_kd_weight():
    T = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(teacher_conf_trust(T, 0.8), torch.tensor([0.2, 1.0, 0.2]))
    a = torch.tensor([0.0, 0.5, 1.0])
    w = kd_weight({"weight": "agreement", "w_min": 0.1}, {"alpha_min": 0.1}, a)
    assert torch.allclose(w, torch.tensor([0.1, 1.0, 0.1]))   # distil most where humans disagree
    assert torch.equal(kd_weight({"weight": "uniform"}, {}, a), torch.ones(3))
