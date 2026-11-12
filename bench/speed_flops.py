"""Efficiency table (step 14): params, FLOPs, FPS of the deployed network, and
training GPU-hours from run logs.

The deployed network is identical for PEdger and ours (same architecture, same
inference code), so params / FLOPs / FPS must come out identical; this script
measures them once on the T4 and once on CPU. Training cost is read from each
run's done.json / log.jsonl; teacher caching hours are added from --teacher-hours.

    python bench/speed_flops.py --size 320 480 --runs runs/* --teacher-hours 2.5 --out results/efficiency.json
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pedger_kd import upstream  # noqa: E402
from pedger_kd.norm import set_norm_mode  # noqa: E402


@torch.no_grad()
def fps(net, x, warmup=20, iters=100):
    for _ in range(warmup):
        net(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        net(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    return iters / (time.time() - t0)


def flops(net, x):
    try:
        from thop import profile
        macs, _ = profile(net, inputs=(x,), verbose=False)
        return {"GMACs": macs / 1e9, "GFLOPs": 2 * macs / 1e9, "tool": "thop"}
    except Exception as e:  # thop missing or failed
        return {"error": f"thop unavailable: {e}"}


def run_cost(run_dir):
    run_dir = Path(run_dir)
    if (run_dir / "done.json").exists():
        return json.load(open(run_dir / "done.json")).get("gpu_hours")
    if (run_dir / "log.jsonl").exists():
        lines = [json.loads(l) for l in open(run_dir / "log.jsonl") if l.strip()]
        return lines[-1]["gpu_hours_total"] if lines else None
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, nargs=2, default=[320, 480])
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--teacher-hours", type=float, default=0.0, help="one-time teacher caching GPU-hours")
    p.add_argument("--cpu-iters", type=int, default=20)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    H, W = args.size
    res = {"input": [H, W], "networks": {}}
    for name, build in (("PEdger deployed (non-recurrent, Guider_noshare)", upstream.build_noshare),
                        ("recurrent (Guider_stu, training only)", upstream.build_recurrent)):
        net = build().eval()
        set_norm_mode(net, "per_sample")
        r = {"params": sum(p.numel() for p in net.parameters())}
        r["params_M"] = round(r["params"] / 1e6, 3)
        r.update(flops(build().eval(), torch.randn(1, 3, H, W)))
        r["fps_cpu"] = round(fps(net, torch.randn(1, 3, H, W), warmup=3, iters=args.cpu_iters), 2)
        if torch.cuda.is_available():
            g = net.cuda()
            r["fps_gpu"] = round(fps(g, torch.randn(1, 3, H, W, device="cuda")), 1)
            r["gpu"] = torch.cuda.get_device_name()
        res["networks"][name] = r
    if args.runs:
        runs = sorted({q for pat in args.runs for q in glob.glob(pat)})
        res["training"] = {Path(r).name: run_cost(r) for r in runs}
        res["teacher_caching_hours"] = args.teacher_hours
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
