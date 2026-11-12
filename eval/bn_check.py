"""The BatchNorm check (step 9).

PEdger's BN layers have no running statistics: they always normalise with the
current batch. PEdger trains and tests with batch 1, so statistics are per image.
Training with B > 1 images per forward pass would pool statistics across images,
and the trained network would then behave differently at test time (batch 1).

Part 1 (--numeric, seconds, CPU or GPU): per-sample normalisation on a batch of B
must equal B separate batch-1 passes; stock BN on a batch of B must not.
Part 2 (--compare-runs): validation scores of short training runs
    configs/checks/bn_micro1.yaml   micro-batch 1 x 16 accumulation (PEdger's setup)
    configs/checks/bn_micro4.yaml   micro-batch 4, per-sample normalisation (our setup)
    configs/checks/bn_stock4.yaml   micro-batch 4, stock BN (what to avoid)
The check passes if micro4 is within --tol of micro1 (seed-level noise). A clear
drop for stock4 shows what the per-sample normalisation protects against.

    python eval/bn_check.py --numeric
    python eval/bn_check.py --compare-runs runs/bn_micro1/scores_val_ns_m.json \
        runs/bn_micro4/scores_val_ns_m.json runs/bn_stock4/scores_val_ns_m.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pedger_kd import upstream  # noqa: E402
from pedger_kd.norm import set_norm_mode  # noqa: E402


@torch.no_grad()
def numeric_check(B=4, H=96, W=128, device="cpu", ckpt=None, seed=0):
    torch.manual_seed(seed)
    out = {}
    for name, build, key in (("noshare", upstream.build_noshare, "G_teacher_noshare"),
                             ("recurrent", upstream.build_recurrent, "G_teacher")):
        net = build().to(device).eval()
        if ckpt:
            net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)[key])
        x = torch.randn(B, 3, H, W, device=device)
        ref = torch.cat([net(x[i:i + 1])[-1] for i in range(B)])      # stock BN, batch 1 each
        stock_b = net(x)[-1]                                           # stock BN, batch B
        set_norm_mode(net, "per_sample")
        ps_b = net(x)[-1]                                              # per-sample, batch B
        out[name] = {"per_sample_vs_batch1_maxabs": float((ps_b - ref).abs().max()),
                     "stock_batchB_vs_batch1_maxabs": float((stock_b - ref).abs().max())}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--numeric", action="store_true")
    p.add_argument("--ckpt", default=None, help="optional: use trained weights (e.g. the released checkpoint)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compare-runs", nargs="+", default=None,
                   help="scores json files; the first is the batch-1 reference")
    p.add_argument("--tol", type=float, default=0.003, help="allowed ODS gap vs the batch-1 run")
    args = p.parse_args()
    ok = True
    if args.numeric:
        res = numeric_check(device=args.device, ckpt=args.ckpt)
        print(json.dumps(res, indent=2))
        for name, r in res.items():
            good = r["per_sample_vs_batch1_maxabs"] < 1e-3
            ok &= good
            print(f"{name}: per-sample batch == batch-1 passes: {'PASS' if good else 'FAIL'}; "
                  f"stock BN batch deviates by {r['stock_batchB_vs_batch1_maxabs']:.3g} (expected > 0)")
    if args.compare_runs:
        recs = [json.load(open(f)) for f in args.compare_runs]
        ref = recs[0]["ODS"]
        print(f"{'run':40s} {'ODS':>7s} {'OIS':>7s} {'gap':>7s}")
        for f, r in zip(args.compare_runs, recs):
            gap = r["ODS"] - ref
            print(f"{Path(f).parent.name:40s} {r['ODS']:7.4f} {r['OIS']:7.4f} {gap:+7.4f}")
        if len(recs) > 1:
            good = abs(recs[1]["ODS"] - ref) <= args.tol
            ok &= good
            print(f"micro-batch run within {args.tol} ODS of batch-1 run: {'PASS' if good else 'FAIL'}")
    if not (args.numeric or args.compare_runs):
        p.error("give --numeric and/or --compare-runs")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
