"""Time training before planning anything else (step 8: "time this run").

Runs a few hundred micro-batches with the real data, real augmentation and the
full step of a late epoch (momentum networks + bilateral augmentation active, so
the estimate is the expensive case), and reports:
  * augmentation-only throughput  -> if close to the full-step throughput, the
    pipeline is input-bound (the GPU waits for data)
  * full train-step throughput, peak GPU memory, average GPU utilisation
  * estimated hours for one full run (samples/epoch x epochs)

    python bench/time_run.py --config configs/baseline.yaml --data-root DATA --teacher-root CACHE
    python bench/time_run.py --config configs/A_agree.yaml ... --set train.micro_batch=8 train.accum_steps=2
"""
import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pedger_kd.config import load_config  # noqa: E402
from pedger_kd.train_kd import Trainer, momentum_update, train_step  # noqa: E402


class GpuUtil(threading.Thread):
    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.samples, self.stop, self.interval = [], False, interval

    def run(self):
        while not self.stop:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                self.samples.append(float(out.split()[0]))
            except Exception:
                return
            time.sleep(self.interval)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data-root", default=".")
    p.add_argument("--teacher-root", default=".")
    p.add_argument("--steps", type=int, default=240, help="micro-batches to time (after warm-up)")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--epoch", type=int, default=20, help="epoch to simulate (>=1 runs the momentum networks)")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    with tempfile.TemporaryDirectory() as tmp:
        tr = Trainer(cfg, tmp, args.data_root, args.teacher_root)
        momentum_update(tr.nets, 1)
        dev = tr.device
        sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
        perm = torch.randperm(len(tr.store))

        def batch(i):
            return tr.next_batch(perm, (i * tr.micro) % max(1, len(perm) - tr.micro))

        for i in range(args.warmup):
            batch(i)
        sync()
        t0 = time.time()
        for i in range(args.steps):
            batch(i)
        sync()
        aug_rate = args.steps * tr.micro / (time.time() - t0)

        for i in range(args.warmup):
            train_step(tr.nets, batch(i), cfg, args.epoch, tr.n_epochs, tr.scaler, tr.accum, tr.amp, tr.bilateral_fn)
        sync()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        mon = GpuUtil()
        if dev.type == "cuda":
            mon.start()
        t0 = time.time()
        for i in range(args.steps):
            train_step(tr.nets, batch(i), cfg, args.epoch, tr.n_epochs, tr.scaler, tr.accum, tr.amp, tr.bilateral_fn)
            if (i + 1) % tr.accum == 0:
                for k in ("rec", "ns"):
                    tr.scaler.step(tr.opts[k])
                tr.scaler.update()
                for k in ("rec", "ns"):
                    tr.opts[k].zero_grad(set_to_none=True)
        sync()
        step_rate = args.steps * tr.micro / (time.time() - t0)
        mon.stop = True

    total = tr.samples_per_epoch * tr.n_epochs
    res = {
        "config": cfg.get("name"), "device": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu",
        "micro_batch": tr.micro, "accum_steps": tr.accum, "amp": tr.amp,
        "augment_only_img_per_s": round(aug_rate, 2), "train_img_per_s": round(step_rate, 2),
        "input_bound": aug_rate < 2 * step_rate,
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev.type == "cuda" else None,
        "gpu_util_mean": round(sum(mon.samples) / len(mon.samples), 1) if mon.samples else None,
        "samples_per_epoch": tr.samples_per_epoch, "epochs": tr.n_epochs,
        "est_hours_full_run": round(total / step_rate / 3600, 2),
    }
    print(json.dumps(res, indent=2))
    print(f"\nOne full run ~ {res['est_hours_full_run']} GPU-h. With ~30 GPU-h/week that is "
          f"~{30 / max(res['est_hours_full_run'], 1e-6):.1f} runs per week.")
    if res["input_bound"]:
        print("Augmentation is not much faster than training: the pipeline may be input-bound.")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
