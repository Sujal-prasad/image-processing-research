"""Sanity test (b), Variant B: with lambda_kd = 0 training is identical to the baseline.
The KD term is still computed (so NaNs in it would show up), only its weight is 0."""
import copy

import pytest
import torch
from conftest import base_cfg, make_batch, needs_pedger


def grads(nets):
    return {f"{k}.{n}": p.grad.clone() for k in ("rec", "ns") for n, p in nets[k].named_parameters()}


def run(cfg, nets, batch, epoch):
    from pedger_kd.train_kd import train_step
    torch.manual_seed(7)
    return train_step(nets, batch, cfg, epoch=epoch, n_epochs=30)


@needs_pedger
@pytest.mark.parametrize("epoch", [0, 5])
def test_lambda_zero_identical_to_baseline(epoch):
    from pedger_kd.train_kd import build_nets, momentum_update
    torch.manual_seed(0)
    nets0 = build_nets("per_sample")
    if epoch >= 1:
        momentum_update(nets0, 1)
    nets1 = copy.deepcopy(nets0)
    batch = make_batch()
    base = base_cfg()
    kd0 = base_cfg(data={"teacher": "x"}, kd={"enabled": True, "lambda": 0.0, "weight": "agreement"})
    l0 = run(base, nets0, batch, epoch)
    l1 = run(kd0, nets1, batch, epoch)
    g0, g1 = grads(nets0), grads(nets1)
    for k in g0:
        assert torch.equal(g0[k], g1[k]), k
    assert torch.isfinite(l1["rec_kd"]) and torch.isfinite(l1["ns_kd"])
    assert torch.equal(l0["rec"], l1["rec"])


@needs_pedger
def test_positive_lambda_changes_gradients():
    from pedger_kd.train_kd import build_nets
    torch.manual_seed(0)
    nets0 = build_nets("per_sample")
    nets1 = copy.deepcopy(nets0)
    batch = make_batch()
    run(base_cfg(), nets0, batch, 0)
    run(base_cfg(data={"teacher": "x"}, kd={"enabled": True, "lambda": 1.0}), nets1, batch, 0)
    g0, g1 = grads(nets0), grads(nets1)
    assert any(not torch.allclose(g0[k], g1[k]) for k in g0)
