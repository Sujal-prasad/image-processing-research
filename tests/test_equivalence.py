"""Speed changes must not change the maths:
* per-image normalisation on a batch == PEdger's batch-1 BatchNorm (the BatchNorm check, numerically)
* per-image loss == the official MyLoss at batch 1
* micro-batch 4 x accumulation 4 == batch 1 x accumulation 16
* the LR schedule is the official one
"""
import copy

import pytest
import torch
from conftest import base_cfg, make_batch, needs_pedger

from pedger_kd.losses import pedger_loss

# Official schedule (GradualWarmupScheduler x8 over 4 epochs + CosineAnnealingLR(30)), measured
# by running PEdger's own scheduler code; note the quirk at epoch 5.
OFFICIAL_LR = [0.001, 0.00275, 0.0045, 0.00625, 0.008, 0.00802197, 0.008, 0.00793432, 0.00782566,
               0.0076752, 0.0074846, 0.00725594, 0.00699173, 0.00669486, 0.00636858, 0.00601648,
               0.0056424, 0.00525045, 0.00484492, 0.00443025, 0.00401099, 0.00359172, 0.00317706,
               0.00277152, 0.00237957, 0.00200549, 0.00165339, 0.00132711, 0.00103024, 0.00076603]


@needs_pedger
@pytest.mark.parametrize("which", ["noshare", "recurrent"])
def test_per_sample_norm_equals_batch1(which):
    from pedger_kd import upstream
    from pedger_kd.norm import set_norm_mode
    torch.manual_seed(0)
    net = (upstream.build_noshare if which == "noshare" else upstream.build_recurrent)()
    x = torch.randn(4, 3, 48, 64)
    with torch.no_grad():
        ref = torch.cat([net(x[i:i + 1])[-1] for i in range(4)])
        stock = net(x)[-1]
        set_norm_mode(net, "per_sample")
        ours = net(x)[-1]
    assert torch.allclose(ours, ref, atol=1e-4), (ours - ref).abs().max()
    assert not torch.allclose(stock, ref, atol=1e-2)   # pooled statistics really do differ


@needs_pedger
def test_per_sample_norm_keeps_state_dict_keys():
    from pedger_kd import upstream
    from pedger_kd.norm import set_norm_mode
    a = upstream.build_noshare()
    keys = list(a.state_dict())
    set_norm_mode(a, "per_sample")
    assert list(a.state_dict()) == keys


@needs_pedger
def test_loss_matches_official_at_batch1_and_averages_over_batch():
    from pedger_kd import upstream
    official = upstream.loss_function().MyLoss()
    torch.manual_seed(0)
    outs = [torch.randn(3, 1, 32, 40) for _ in range(11)]
    b = make_batch(B=3, H=32, W=40)
    soft = 0.7 * b["Y"] + 0.3 * torch.rand_like(b["Y"])
    per_img = []
    for i in range(3):
        ref, _ = official([torch.sigmoid(o[i:i + 1]) for o in outs], b["Y"][i:i + 1], soft[i:i + 1])
        ours = pedger_loss([o[i:i + 1] for o in outs], b["Y"][i:i + 1], soft[i:i + 1])
        assert torch.allclose(ours, ref, atol=1e-6)
        per_img.append(ref)
    assert torch.allclose(pedger_loss(outs, b["Y"], soft), torch.stack(per_img).mean(), atol=1e-6)


@needs_pedger
@pytest.mark.parametrize("epoch", [0, 3])
def test_microbatch_accumulation_equals_batch1_accumulation(epoch):
    """Checked in float64, where mathematically identical computations agree to ~1e-12.
    (In float32 the toy 32x48 inputs leave 2x3-pixel maps at the deepest recurrent step, where
    per-image variance is tiny and rounding is amplified; at real image sizes the float32 gap is ~1e-6.)
    Epoch 3 includes the momentum networks and the soft target; bilateral augmentation is off
    because it draws random numbers per micro-batch."""
    from pedger_kd.train_kd import build_nets, momentum_update, train_step
    torch.manual_seed(0)
    nets1 = build_nets("per_sample")
    for k in nets1:
        nets1[k].double()
    if epoch:
        momentum_update(nets1, 1)
    nets4 = copy.deepcopy(nets1)
    cfg = base_cfg()
    cfg["train"]["bilateral"] = False
    full = {k: (v.double() if torch.is_tensor(v) else v) for k, v in make_batch(B=16, H=32, W=48).items()}
    for i in range(16):   # batch 1 x 16 (official)
        train_step(nets1, {k: (v[i:i + 1] if torch.is_tensor(v) else v) for k, v in full.items()}, cfg, epoch, 30, accum=16)
    for i in range(0, 16, 4):  # micro-batch 4 x 4
        train_step(nets4, {k: (v[i:i + 4] if torch.is_tensor(v) else v) for k, v in full.items()}, cfg, epoch, 30, accum=4)
    for k in ("rec", "ns"):
        gmax = max(p.grad.abs().max() for p in nets1[k].parameters())
        for (n, p1), (_, p4) in zip(nets1[k].named_parameters(), nets4[k].named_parameters()):
            assert (p1.grad - p4.grad).abs().max() <= 1e-10 * gmax, f"{k}.{n}"


@needs_pedger
def test_lr_schedule_is_official():
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from pedger_kd import upstream
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1e-3)
    warm = upstream.utils().GradualWarmupScheduler(opt, multiplier=8, total_epoch=4,
                                                   after_scheduler=CosineAnnealingLR(opt, 30))
    lrs = []
    for _ in range(30):
        lrs.append(opt.param_groups[0]["lr"])
        warm.step()
    assert lrs == pytest.approx(OFFICIAL_LR, rel=1e-5)


@needs_pedger
def test_parameter_counts():
    from pedger_kd import upstream
    assert sum(p.numel() for p in upstream.build_noshare().parameters()) == 734225    # the 0.73M model
    assert sum(p.numel() for p in upstream.build_recurrent().parameters()) == 364685
