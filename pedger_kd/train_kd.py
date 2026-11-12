"""PEdger training loop with minimal hooks for Variant A (target) and Variant B (loss).

Faithful to the official main.py:
  * two students (recurrent Guider_stu, non-recurrent Guider_noshare) and their
    momentum copies; copies are set at epoch 1 and averaged W_m <- (W + W_m)/2 at
    the start of every later epoch
  * soft target from the momentum pair from epoch 1 on, eta = 0.8 * ep / EP
  * bilateral smoothing / sharpening of the input with probability eta
  * AdamW(lr 1e-3, betas (0.9, 0.9), wd 1e-3); the official GradualWarmupScheduler
    (x8 over 4 epochs) followed by CosineAnnealingLR(T_max = EP), stepped per epoch
  * effective batch 16 per optimizer step, 30 epochs
Changes for speed only (numerically equivalent, see tests/):
  * micro-batches of B images with per-image normalisation and per-image loss
    instead of batch 1, accumulated to 16 images per step
  * data and augmentation on the GPU, mixed precision (fp16 on T4)

Run:
    python -m pedger_kd.train_kd --config configs/A_agree.yaml --seed 0 \
        --data-root /kaggle/input/pedger-data --teacher-root /kaggle/input/pedger-teachers \
        --out runs/A_agree_s0
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from . import upstream
from .augment import JointAugment
from .config import dump_config, load_config
from .data import ArrayStore, read_meta
from .kd_loss import kd_loss, kd_weight
from .losses import pedger_loss
from .norm import set_norm_mode
from .soft_target import build_target, fuse_momentum, pedger_eta

NET_KEYS = {"rec": "G", "rec_m": "G_teacher", "ns": "G_noshare", "ns_m": "G_teacher_noshare"}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_nets(norm="per_sample", device="cpu", channels_last=False):
    nets = {"rec": upstream.build_recurrent(), "ns": upstream.build_noshare()}
    nets["rec_m"] = upstream.build_recurrent()
    nets["ns_m"] = upstream.build_noshare()
    for k, n in nets.items():
        set_norm_mode(n, norm)
        n.to(device)
        if channels_last:
            n.to(memory_format=torch.channels_last)
    for k in ("rec_m", "ns_m"):
        for p in nets[k].parameters():
            p.requires_grad = False
    return nets


def momentum_update(nets, epoch):
    """Official rule: copy at epoch 1, average at every epoch >= 2 (start of epoch)."""
    for s, m in (("rec", "rec_m"), ("ns", "ns_m")):
        if epoch == 1:
            nets[m].load_state_dict(nets[s].state_dict())
        elif epoch >= 2:
            st, sm = nets[s].state_dict(), nets[m].state_dict()
            nets[m].load_state_dict({k: (sm[k] + st[k]) * 0.5 for k in sm})


def bilateral_aug(img, eta, bilateral_fn):
    """Per image, with probability eta: bilateral-smoothed or sharpened input (official main.py)."""
    B = img.shape[0]
    pick = torch.rand(B, device=img.device) < eta
    if not pick.any():
        return img
    smooth = bilateral_fn(img.float(), 5).to(img.dtype)
    sharpen_choice = torch.rand(B, device=img.device) > 0.5
    aug = torch.where(sharpen_choice[:, None, None, None], smooth, img + 2 * (img - smooth))
    return torch.where(pick[:, None, None, None], aug, img)


def train_step(nets, batch, cfg, epoch, n_epochs, scaler=None, accum=1, amp=False, bilateral_fn=None):
    """Forward/backward for one micro-batch (no optimizer step). Returns detached losses."""
    method, kd = cfg.get("method", {}), cfg.get("kd", {})
    img, Y, a_alpha, T = batch["img"], batch["Y"], batch["a_alpha"], batch["T"]
    dev = img.device.type
    eta = pedger_eta(epoch, n_epochs)

    M = None
    if epoch >= 1:
        with torch.no_grad(), torch.autocast(device_type=dev, enabled=amp):
            z_r = nets["rec_m"](img)[-1]
            z_n = nets["ns_m"](img)[-1]
        M = fuse_momentum(z_r.float(), z_n.float())
    target = build_target(method, Y, M, T, a_alpha, epoch, n_epochs)

    if bilateral_fn is not None and cfg["train"].get("bilateral", True):
        img = bilateral_aug(img, eta, bilateral_fn)

    out = {}
    for name in ("rec", "ns"):
        with torch.autocast(device_type=dev, enabled=amp):
            logits = nets[name](img)
        loss = pedger_loss(logits, Y, target)
        if kd.get("enabled", False):
            if T is None:
                raise ValueError("kd.enabled needs data.teacher")
            w = kd_weight(kd, method, a_alpha)
            lkd = kd_loss(logits[-1], T, w, tau=kd.get("tau", 2.0), use_ls=kd.get("use_ls", True))
            loss = loss + kd.get("lambda", 1.0) * lkd
            out[f"{name}_kd"] = lkd.detach()
        scaled = loss / accum
        (scaler.scale(scaled) if scaler is not None else scaled).backward()
        out[name] = loss.detach()
    return out


class Trainer:
    def __init__(self, cfg, out_dir, data_root=None, teacher_root=None, device=None, max_hours=None):
        self.cfg = cfg
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_hours = max_hours
        self._session_start = time.time()
        tc = cfg["train"]
        seed_everything(int(cfg.get("seed", 0)))
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.amp = bool(tc.get("amp", True)) and self.device.type == "cuda"
        self.channels_last = bool(tc.get("channels_last", True)) and self.device.type == "cuda"
        self.nets = build_nets(tc.get("norm", "per_sample"), self.device, self.channels_last)
        self.n_epochs = int(tc.get("epochs", 30))
        self.micro = int(tc.get("micro_batch", 4))
        self.accum = int(tc.get("accum_steps", 4))
        if tc.get("norm", "per_sample") == "stock" and self.micro > 1:
            print("WARNING: stock BatchNorm with micro_batch > 1 pools statistics across images "
                  "(this is the configuration the BatchNorm check is meant to catch).")

        lr = float(tc.get("lr", 1e-3))
        U = upstream.utils()
        self.opts, self.scheds = {}, {}
        for k in ("rec", "ns"):
            opt = torch.optim.AdamW([p for p in self.nets[k].parameters() if p.requires_grad],
                                    lr=lr, betas=(0.9, 0.9), weight_decay=1e-3)
            cos = CosineAnnealingLR(opt, self.n_epochs)
            warm = U.GradualWarmupScheduler(opt, multiplier=tc.get("warmup_multiplier", 8),
                                            total_epoch=tc.get("warmup_epochs", 4), after_scheduler=cos)
            self.opts[k], self.scheds[k] = opt, (warm, cos)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.bilateral_fn = U.bilateralFilter

        dc = cfg["data"]
        store_dir = Path(data_root or ".") / dc["store"]
        tdir = Path(teacher_root or ".") / dc.get("teacher_dir", dc["store"]) if dc.get("teacher") else None
        self.meta = read_meta(store_dir)
        self.store = ArrayStore(store_dir, dc["split"], tdir, dc.get("teacher"),
                                dilation=int(dc.get("agreement_dilation", 0)),
                                device=self.device, on_gpu=dc.get("data_on_gpu", True))
        self.aug = JointAugment(cfg.get("augment", {}), train=True)
        spe = dc.get("samples_per_epoch") or len(self.store) * self.aug.num_variants()
        step = self.micro * self.accum
        self.samples_per_epoch = int(math.ceil(spe / step) * step)
        self.micro_per_epoch = self.samples_per_epoch // self.micro

        self.start_epoch, self.elapsed = 0, 0.0
        self._resume()
        dump_config(cfg, self.out / "config.yaml")

    # ---------------------------------------------------------------- state
    def state(self, epoch):
        st = {NET_KEYS[k]: n.state_dict() for k, n in self.nets.items()}
        st.update({
            "optimizer": self.opts["rec"].state_dict(), "optimizer_noshare": self.opts["ns"].state_dict(),
            "scheduler_warmup": self.scheds["rec"][0].state_dict(), "scheduler": self.scheds["rec"][1].state_dict(),
            "scheduler_warmup_noshare": self.scheds["ns"][0].state_dict(),
            "scheduler_noshare": self.scheds["ns"][1].state_dict(),
            "scaler": self.scaler.state_dict(), "epoch": epoch, "elapsed_sec": self.elapsed,
            "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
            "config": self.cfg,
        })
        return st

    def _resume(self):
        path = self.out / "last.pt"
        if not path.exists():
            return
        st = torch.load(path, map_location=self.device, weights_only=False)
        for k, n in self.nets.items():
            n.load_state_dict(st[NET_KEYS[k]])
        self.opts["rec"].load_state_dict(st["optimizer"])
        self.opts["ns"].load_state_dict(st["optimizer_noshare"])
        self.scheds["rec"][0].load_state_dict(st["scheduler_warmup"])
        self.scheds["rec"][1].load_state_dict(st["scheduler"])
        self.scheds["ns"][0].load_state_dict(st["scheduler_warmup_noshare"])
        self.scheds["ns"][1].load_state_dict(st["scheduler_noshare"])
        self.scaler.load_state_dict(st["scaler"])
        rng = st["rng"]
        torch.set_rng_state(rng["torch"].cpu())
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([r.cpu() for r in rng["cuda"]])
        self.start_epoch = int(st["epoch"]) + 1
        self.elapsed = float(st.get("elapsed_sec", 0.0))
        print(f"Resumed from {path} at epoch {self.start_epoch}")

    def save(self, epoch):
        tmp = self.out / "last.pt.tmp"
        torch.save(self.state(epoch), tmp)
        os.replace(tmp, self.out / "last.pt")
        if self.cfg["train"].get("keep_epoch_ckpts", False):
            torch.save({NET_KEYS[k]: n.state_dict() for k, n in self.nets.items()},
                       self.out / f"epoch_{epoch:02d}.pt")

    # ---------------------------------------------------------------- loop
    def next_batch(self, perm, pos):
        idx = perm[pos:pos + self.micro]
        if len(idx) < self.micro:
            idx = torch.cat([idx, torch.randperm(len(self.store))[: self.micro - len(idx)]])
        imgs, frac, frac_a, T = self.store.get(idx)
        b = self.aug(imgs, frac, frac_a, T)
        if self.channels_last:
            b["img"] = b["img"].contiguous(memory_format=torch.channels_last)
        return b

    def run_epoch(self, epoch):
        momentum_update(self.nets, epoch)
        for n in self.nets.values():
            n.train()
        sums, count = {}, 0
        perm, pos = torch.randperm(len(self.store)), 0
        t0 = time.time()
        log_every = int(self.cfg["train"].get("log_every", 200))
        for i in range(self.micro_per_epoch):
            if pos >= len(perm):
                perm, pos = torch.randperm(len(self.store)), 0
            batch = self.next_batch(perm, pos)
            pos += self.micro
            losses = train_step(self.nets, batch, self.cfg, epoch, self.n_epochs, self.scaler,
                                self.accum, self.amp, self.bilateral_fn)
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v
            count += 1
            if (i + 1) % self.accum == 0:
                for k in ("rec", "ns"):
                    self.scaler.step(self.opts[k])
                self.scaler.update()
                for k in ("rec", "ns"):
                    self.opts[k].zero_grad(set_to_none=True)
            if (i + 1) % log_every == 0:
                cur = {k: float(v) / count for k, v in sums.items()}
                if not all(math.isfinite(v) for v in cur.values()):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch} step {i}: {cur}")
                rate = (i + 1) * self.micro / (time.time() - t0)
                eta_min = (self.samples_per_epoch - (i + 1) * self.micro) / rate / 60
                print(f"[ep {epoch}/{self.n_epochs} {i + 1}/{self.micro_per_epoch}] "
                      + " ".join(f"{k}={v:.4f}" for k, v in cur.items())
                      + f" | {rate:.1f} img/s, epoch ETA {eta_min:.1f} min", flush=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        secs = time.time() - t0
        means = {k: float(v) / max(count, 1) for k, v in sums.items()}
        return secs, means

    def run(self):
        print(f"Training {self.cfg.get('name')} seed={self.cfg.get('seed')} on {self.device}: "
              f"{self.samples_per_epoch} samples/epoch x {self.n_epochs} epochs, "
              f"micro_batch={self.micro} x accum={self.accum}, amp={self.amp}", flush=True)
        for epoch in range(self.start_epoch, self.n_epochs):
            if self.max_hours is not None and epoch > self.start_epoch:
                per_epoch = self.elapsed / max(epoch, 1)
                if (time.time() - self._session_start) / 3600 + per_epoch / 3600 > self.max_hours:
                    print(f"Stopping before epoch {epoch}: --max-hours {self.max_hours} would be exceeded. "
                          f"Re-run the same command to resume.")
                    return False
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            lr = self.opts["rec"].param_groups[0]["lr"]
            secs, means = self.run_epoch(epoch)
            self.elapsed += secs
            for k in ("rec", "ns"):
                self.scheds[k][0].step()
            rec = {"epoch": epoch, "lr": lr, "seconds": round(secs, 1), "gpu_hours_total": self.elapsed / 3600,
                   "samples": self.samples_per_epoch, **means}
            if self.device.type == "cuda":
                rec["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                rec["gpu"] = torch.cuda.get_device_name()
            with open(self.out / "log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            self.save(epoch)
        torch.save({NET_KEYS[k]: n.state_dict() for k, n in self.nets.items()}, self.out / "final.pt")
        with open(self.out / "done.json", "w") as f:
            json.dump({"gpu_hours": self.elapsed / 3600, "epochs": self.n_epochs,
                       "samples_per_epoch": self.samples_per_epoch}, f, indent=2)
        print(f"Done: {self.out / 'final.pt'} ({self.elapsed / 3600:.2f} GPU-h)")
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--data-root", default=".")
    p.add_argument("--teacher-root", default=".")
    p.add_argument("--max-hours", type=float, default=None,
                   help="stop cleanly before an epoch that would exceed this session budget")
    p.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=10")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg["seed"] = args.seed
    Trainer(cfg, args.out, args.data_root, args.teacher_root, max_hours=args.max_hours).run()


if __name__ == "__main__":
    main()
