"""Sanity test (a), Variant A: gamma = 0 and alpha = 1 - eta must reproduce PEdger's
original soft target, both as a formula and through the full training step."""
import copy

import torch
from conftest import base_cfg, make_batch, needs_pedger

from pedger_kd.soft_target import build_target, fuse_momentum, pedger_eta

REDUCED = {"target": "agreement", "alpha_mode": "eta", "gamma": 0.0, "ramp": False}


def test_formula_reduces_to_pedger_every_epoch():
    b = make_batch(B=3)
    M = torch.rand_like(b["Y"])
    for ep in range(30):
        ref = build_target({"target": "pedger"}, b["Y"], M if ep >= 1 else None, b["T"], b["a_alpha"], ep, 30)
        ours = build_target(REDUCED, b["Y"], M if ep >= 1 else None, b["T"], b["a_alpha"], ep, 30)
        assert torch.allclose(ours, ref, atol=1e-7, rtol=0), f"epoch {ep}: max diff {(ours - ref).abs().max()}"
    # epoch 0: exactly the label
    assert torch.equal(build_target(REDUCED, b["Y"], None, b["T"], b["a_alpha"], 0, 30), b["Y"])


def test_matches_official_main_py_expression():
    """Official main.py: edge_gt_soft = edge_gt * (1 - dis_weight) + res * dis_weight."""
    torch.manual_seed(0)
    zt, zn, gt = torch.randn(2, 1, 20, 20), torch.randn(2, 1, 20, 20), (torch.rand(2, 1, 20, 20) > 0.9).float()
    unc = torch.abs(torch.sigmoid(zt) - 0.5)
    unc_n = torch.abs(torch.sigmoid(zn) - 0.5)
    weight = unc / (unc + unc_n)
    res = torch.sigmoid(zt * weight + zn * (1 - weight))
    dis = 0.8 * 7 / 30
    official = gt * (1 - dis) + res * dis
    M = fuse_momentum(zt, zn)
    assert torch.allclose(M, res, atol=1e-7)
    ours = build_target(REDUCED, gt, M, None, gt, 7, 30)
    assert torch.allclose(ours, official, atol=1e-6)
    assert abs(pedger_eta(7, 30) - dis) < 1e-12


@needs_pedger
def test_training_step_identical_to_baseline():
    from pedger_kd.train_kd import build_nets, momentum_update, train_step
    base = base_cfg()
    red = copy.deepcopy(base)
    red["method"].update(REDUCED)
    torch.manual_seed(0)
    nets_a = build_nets("per_sample")
    for ep in (1, 2):
        momentum_update(nets_a, ep)
    nets_b = copy.deepcopy(nets_a)
    batch = make_batch(B=2, H=48, W=64)
    for nets, cfg in ((nets_a, base), (nets_b, red)):
        torch.manual_seed(123)
        train_step(nets, batch, cfg, epoch=9, n_epochs=30)
    for k in ("rec", "ns"):
        for (n, pa), (_, pb) in zip(nets_a[k].named_parameters(), nets_b[k].named_parameters()):
            assert torch.allclose(pa.grad, pb.grad, atol=1e-6, rtol=1e-5), f"{k}.{n}"
