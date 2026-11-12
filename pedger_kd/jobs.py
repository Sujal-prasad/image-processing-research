"""Unattended, resumable training across Kaggle sessions and Kaggle accounts.

A queue file lists jobs (config + seed) in priority order. Every session (any
account) runs:

    python -m pedger_kd.jobs --queue queues/main.yaml --remote hf:<user>/<repo> --ws /kaggle/working/ws \
        --hours 11.3 --gpus auto

and for each job, in order:
  1. skip it if the remote says it is done, failed, or being trained right now by
     another session (heartbeat younger than --stale-minutes)
  2. claim it (write progress.json with this worker's id, re-read to win races)
  3. pull its run folder (last.pt etc.), resume exactly where it stopped
  4. train; every --ckpt-minutes and at every epoch end the checkpoint +
     progress.json are uploaded in the background
  5. at the session deadline: stop cleanly, upload, mark "stopped" (immediately
     claimable by the next session); when finished: upload final.pt, mark "done"
A hard kill (quota exhausted, crash) loses at most --ckpt-minutes of work; the
job becomes claimable again once its heartbeat is older than --stale-minutes.
With 2 GPUs (--gpus auto on Kaggle's 2xT4) two workers run different jobs in parallel.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import yaml

from .config import load_config
from .sync import AsyncPusher, age_minutes, iso, open_remote

RUN_FILES = ["last.pt", "log.jsonl", "config.yaml", "progress.json"]
FINAL_FILES = ["final.pt", "done.json", "last.pt", "log.jsonl", "config.yaml", "progress.json"]


def load_queue(path):
    with open(path) as f:
        q = yaml.safe_load(f)
    jobs = []
    for idx, e in enumerate(q["jobs"]):
        seeds = e.get("seeds", [e.get("seed", 0)])
        for s in seeds:
            cfg = load_config(e["config"], e.get("set", []))
            jobs.append({"config": e["config"], "seed": int(s), "set": list(e.get("set", [])),
                         "name": f"{cfg['name']}_s{s}", "_order": (s, idx) if q.get("order") == "seed_major" else (idx, s)})
    jobs.sort(key=lambda j: j["_order"])
    names = [j["name"] for j in jobs]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"duplicate job names in {path}: {dup} (give each job a unique config name)")
    return jobs


def needed_prefixes(jobs):
    out = set()
    for j in jobs:
        cfg = load_config(j["config"], j["set"])
        out.add(f"data/{cfg['data']['store']}")
        if cfg["data"].get("teacher"):
            out.add(f"teachers/{cfg['data'].get('teacher_dir', cfg['data']['store'])}")
    return sorted(out)


class Worker:
    def __init__(self, remote, ws, worker_id, deadline, data_root=None, teacher_root=None, stale_minutes=75,
                 ckpt_minutes=30, settle_seconds=15, take_over=False, retry_failed=False, device=None):
        self.remote, self.ws, self.id, self.deadline = remote, Path(ws), worker_id, deadline
        self.data_root = Path(data_root) if data_root else self.ws / "data"
        self.teacher_root = Path(teacher_root) if teacher_root else self.ws / "teachers"
        self.stale, self.ckpt_minutes, self.settle = stale_minutes, ckpt_minutes, settle_seconds
        self.take_over, self.retry_failed, self.device = take_over, retry_failed, device
        self.trainer_kwargs = {}   # extra Trainer arguments (tests inject should_stop here)

    # --------------------------------------------------------------- progress / claim
    def _progress(self, name, status, trainer=None, extra=None):
        run_dir = self.ws / "runs" / name
        run_dir.mkdir(parents=True, exist_ok=True)
        p = {"run": name, "status": status, "worker": self.id, "heartbeat": iso()}
        if trainer is not None:
            p.update({"epoch": trainer.position["epoch"], "step": trainer.position["step"],
                      "epochs": trainer.n_epochs, "micro_steps_per_epoch": trainer.micro_per_epoch,
                      "gpu_hours": round(trainer.elapsed / 3600, 3), "session": trainer.sessions})
        p.update(extra or {})
        path = run_dir / "progress.json"
        path.write_text(json.dumps(p, indent=2))
        return path

    def claim(self, name):
        p = self.remote.read_json(f"runs/{name}/progress.json")
        if p:
            st = p.get("status")
            if st == "done":
                return "done"
            if st == "failed" and not self.retry_failed:
                return "failed-before"
            if (st == "running" and p.get("worker") != self.id and not self.take_over
                    and age_minutes(p.get("heartbeat")) < self.stale):
                return "busy"
        path = self._progress(name, "running", extra={"claimed": iso()})
        self.remote.upload({f"runs/{name}/progress.json": str(path)}, f"claim {name} by {self.id}")
        time.sleep(self.settle)
        p2 = self.remote.read_json(f"runs/{name}/progress.json") or {}
        return "claimed" if p2.get("worker") == self.id else "busy"

    # --------------------------------------------------------------- one job
    def run_job(self, job):
        from .train_kd import Trainer
        name = job["name"]
        state = self.claim(name)
        if state != "claimed":
            print(f"[{self.id}] skip {name}: {state}", flush=True)
            return state
        run_dir = self.ws / "runs" / name
        n = self.remote.download(f"runs/{name}", self.ws)
        print(f"[{self.id}] === {name} ({job['config']}, seed {job['seed']}); pulled {n} files", flush=True)
        self._progress(name, "running", extra={"claimed": iso()})   # pulled copy may be older than our claim
        pusher = AsyncPusher(self.remote, self.ws / ".staging" / name)
        cfg = load_config(job["config"], job["set"])
        cfg["seed"] = job["seed"]
        trainer = None
        pushed = {"done": False}
        try:
            trainer = Trainer(cfg, run_dir, self.data_root, self.teacher_root, device=self.device,
                              deadline=self.deadline, ckpt_minutes=self.ckpt_minutes, **self.trainer_kwargs)

            def on_checkpoint(kind):
                status = {"periodic": "running", "epoch": "running", "stopped": "stopped", "done": "done"}[kind]
                self._progress(name, status, trainer)
                files = FINAL_FILES if kind == "done" else RUN_FILES
                pusher.request({f"runs/{name}/{f}": str(run_dir / f) for f in files if (run_dir / f).exists()},
                               f"{name}: {kind} epoch {trainer.position['epoch']} step {trainer.position['step']}")
                if kind in ("stopped", "done"):
                    pusher.flush()
                pushed["done"] |= kind == "done"

            trainer.on_checkpoint = on_checkpoint
            finished = trainer.run()
            if finished and not pushed["done"]:
                on_checkpoint("done")       # finished earlier but never marked done on the remote
            return "done" if finished else "stopped"
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[{self.id}] FAILED {name}:\n{tb}", flush=True)
            path = self._progress(name, "failed", trainer, extra={"error": f"{type(e).__name__}: {e}",
                                                                  "traceback": tb[-4000:]})
            pusher.request({f"runs/{name}/progress.json": str(path)}, f"{name}: failed")
            return "failed"
        finally:
            if not pusher.flush():
                print(f"[{self.id}] WARNING: final upload for {name} did not finish", flush=True)

    def run(self, jobs, min_minutes_left=15):
        summary = {}
        for job in jobs:
            left = (self.deadline - time.time()) / 60
            if left < min_minutes_left:
                print(f"[{self.id}] {left:.0f} min left in this session: stopping.", flush=True)
                break
            summary[job["name"]] = self.run_job(job)
            if summary[job["name"]] == "stopped":
                break
        return summary


def _stream(proc, prefix):
    for line in iter(proc.stdout.readline, ""):
        print(f"{prefix} {line}", end="", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queue", required=True)
    p.add_argument("--remote", default=os.environ.get("PEDGER_REMOTE"), help="hf:<user>/<repo> or local:<dir>")
    p.add_argument("--ws", default="ws", help="local workspace (mirrors the remote)")
    p.add_argument("--hours", type=float, default=11.3, help="session budget, counted from --session-start")
    p.add_argument("--session-start", type=float, default=None, help="unix time the session started")
    p.add_argument("--gpus", default="auto", help="'auto', 'none' or e.g. '0,1'")
    p.add_argument("--label", default="", help="free text identifying the account/session in progress.json")
    p.add_argument("--data-root", default=None, help="use an attached copy instead of pulling data/")
    p.add_argument("--teacher-root", default=None, help="use an attached copy instead of pulling teachers/")
    p.add_argument("--ckpt-minutes", type=float, default=30.0)
    p.add_argument("--stale-minutes", type=float, default=75.0)
    p.add_argument("--settle-seconds", type=float, default=15.0)
    p.add_argument("--take-over", action="store_true", help="ignore other workers' locks (only if sure they are dead)")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--squash", action="store_true", help="squash remote history at the end (saves HF storage)")
    p.add_argument("--worker-gpu", default=None, help=argparse.SUPPRESS)
    p.add_argument("--deadline", type=float, default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    remote = open_remote(args.remote)
    if remote is None:
        p.error("--remote (or $PEDGER_REMOTE) is required")
    jobs = load_queue(args.queue)
    ws = Path(args.ws)
    deadline = args.deadline or (args.session_start or time.time()) + args.hours * 3600

    if args.worker_gpu is not None:      # child process: one worker on one GPU
        dev = "cpu" if args.worker_gpu == "cpu" else f"gpu{args.worker_gpu}"
        wid = f"{args.label or socket.gethostname()}:{dev}:{os.getpid()}"
        summary = Worker(remote, ws, wid, deadline, args.data_root, args.teacher_root, args.stale_minutes,
                         args.ckpt_minutes, args.settle_seconds, args.take_over, args.retry_failed).run(jobs)
        print(f"[{wid}] summary: {json.dumps(summary)}", flush=True)
        return

    print(f"queue {args.queue}: {len(jobs)} jobs; remote {remote.spec}; "
          f"session ends in {(deadline - time.time()) / 3600:.2f} h", flush=True)
    for prefix in needed_prefixes(jobs):
        kind, name = prefix.split("/", 1)
        root = args.data_root if kind == "data" else args.teacher_root
        if root and (Path(root) / name).exists():
            print(f"using attached {Path(root) / name}")
            continue
        if (ws / prefix).exists() and any((ws / prefix).rglob("*.npy")):
            print(f"{prefix}: already in workspace")
            continue
        t0 = time.time()
        n = remote.download(prefix, ws)
        print(f"pulled {prefix}: {n} files in {time.time() - t0:.0f}s", flush=True)
        if n == 0:
            print(f"WARNING: {prefix} is not on the remote yet; jobs that need it will fail and be marked failed")

    if args.gpus == "none":
        gpus = []
    elif args.gpus == "auto":
        try:
            import torch
            gpus = [str(i) for i in range(torch.cuda.device_count())]
        except Exception:
            gpus = []
    else:
        gpus = [g for g in args.gpus.split(",") if g != ""]
    gpus = gpus or ["cpu"]
    print(f"workers: {gpus}", flush=True)

    procs, threads = [], []
    base = [sys.executable, "-m", "pedger_kd.jobs", "--queue", args.queue, "--remote", remote.spec, "--ws", str(ws),
            "--deadline", str(deadline), "--label", args.label, "--ckpt-minutes", str(args.ckpt_minutes),
            "--stale-minutes", str(args.stale_minutes), "--settle-seconds", str(args.settle_seconds)]
    for opt in ("data_root", "teacher_root"):
        if getattr(args, opt):
            base += [f"--{opt.replace('_', '-')}", getattr(args, opt)]
    base += ["--take-over"] if args.take_over else []
    base += ["--retry-failed"] if args.retry_failed else []
    for i, g in enumerate(gpus):
        env = dict(os.environ)
        if g != "cpu":
            env["CUDA_VISIBLE_DEVICES"] = g
        proc = subprocess.Popen(base + ["--worker-gpu", g], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        t = threading.Thread(target=_stream, args=(proc, "[cpu]" if g == "cpu" else f"[gpu{g}]"), daemon=True)
        t.start()
        procs.append(proc)
        threads.append(t)
        if i + 1 < len(gpus):
            time.sleep(args.settle_seconds + 15)   # stagger claims so workers pick different jobs
    codes = [proc.wait() for proc in procs]
    for t in threads:
        t.join(timeout=10)
    if args.squash:
        try:
            remote.squash()
            print("remote history squashed")
        except Exception as e:
            print(f"squash failed (harmless): {e}")
    print(f"workers exited with {codes}")


if __name__ == "__main__":
    main()
