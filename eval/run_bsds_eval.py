"""Predict (optional) and score ODS / OIS / AP on a split (tasks 1 and 10).

Score a trained run (writes <run>/scores_<split>_<mode>[_ms].json for make_tables.py):
    python eval/run_bsds_eval.py --run runs/A_agree_s0 --store data_store/bsds --split test \
        --raw-root BSR/BSDS500/data
Step 1, the released PEdger checkpoint (eval/released_gate.py runs this for every mode):
    python eval/run_bsds_eval.py --ckpt third_party/PEdger/checkpoint_PEdger.pth --mode ns_m \
        --store data_store/bsds --split test --raw-root BSR/BSDS500/data --out-dir outputs/released/ns_m
Score an existing folder of PNG predictions (e.g. a teacher):
    python eval/run_bsds_eval.py --pred-dir work/test_sauge --store data_store/bsds --split test --raw-root ...

Resumable across Kaggle sessions with --sync-to <remote folder>: the remote folder mirrors the output
folder (the run folder, or --out-dir / --pred-dir). Predictions are uploaded once made, scoring progress
every 10 minutes, the scores at the end; a later call skips what is already there. On Kaggle the script
stops before the 12-hour session limit and exits with code 3 ("stopped for time, run again").
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pedger_kd import session  # noqa: E402
from pedger_kd.augment import normalize_image_u8  # noqa: E402
from pedger_kd.config import load_config  # noqa: E402
from pedger_kd.data import read_meta  # noqa: E402
from pedger_kd.evaluation import ScoringStopped, cache_path, evaluate, read_eval_list  # noqa: E402
from pedger_kd.inference import load_nets, predict  # noqa: E402
from pedger_kd.sync import open_remote  # noqa: E402

STOPPED_FOR_TIME = 3
PRED_MARKER = "PREDICTIONS_DONE.json"


def write_predictions(ckpt, items, out_dir, mode, transform, scales, device):
    nets = load_nets(ckpt, device=device, mode=mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    step = max(1, len(items) // 10)
    for i, it in enumerate(items, 1):
        img = cv2.cvtColor(cv2.imread(it["image"]), cv2.COLOR_BGR2RGB)
        prob = predict(nets, normalize_image_u8(img, device), mode, transform, scales)
        cv2.imwrite(str(out_dir / f"{it['name']}.png"), (prob.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy())
        if i % step == 0 or i == len(items):
            el = time.time() - t0
            print(f"  predicted {i}/{len(items)} | {el / 60:.1f} min so far | about "
                  f"{el / i * (len(items) - i) / 60:.0f} min left", flush=True)
    return time.time() - t0


class Mirror:
    """Keeps a local folder and a remote folder in step (no-op without a remote)."""

    def __init__(self, remote, prefix, local_root):
        self.remote, self.prefix, self.root = remote, (prefix or "").strip("/"), Path(local_root)

    def rp(self, local_path):
        rel = Path(local_path).resolve().relative_to(self.root.resolve()).as_posix()
        return self.prefix if rel == "." else f"{self.prefix}/{rel}"

    def exists(self, local_path):
        return self.remote is not None and self.remote.exists(self.rp(local_path))

    def read_json(self, local_path):
        return self.remote.read_json(self.rp(local_path)) if self.remote is not None else None

    def pull(self, local_path):
        """Download the remote copy of a file or folder to local_path."""
        if self.remote is None:
            return 0
        with tempfile.TemporaryDirectory() as tmp:
            n = self.remote.download(self.rp(local_path), tmp)
            src = Path(tmp) / self.rp(local_path)
            if src.is_file():
                Path(local_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, local_path)
            elif src.is_dir():
                shutil.copytree(src, local_path, dirs_exist_ok=True)
        return n

    def push(self, paths, message):
        if self.remote is None:
            return
        files = {self.rp(p): str(p) for p in paths if Path(p).is_file()}
        if files:
            self.remote.upload(files, message)


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="training run directory (uses final.pt or last.pt)")
    g.add_argument("--ckpt")
    g.add_argument("--pred-dir")
    p.add_argument("--store", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--raw-root", required=True)
    p.add_argument("--mode", default=None, help="ns_m (default from config) | rec_m | fuse_m | ns | rec | fuse")
    p.add_argument("--transform", default=None, help="pedger (default) | sigmoid")
    p.add_argument("--ms", action="store_true", help="multi-scale 0.5/1/1.5")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--nproc", type=int, default=0, help="scoring processes (0 = number of CPUs)")
    p.add_argument("--thresholds", type=int, default=99)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--predict-only", action="store_true")
    p.add_argument("--sync-to", default=None, help="remote folder mirroring the output folder (resume + save)")
    p.add_argument("--remote", default=os.environ.get("PEDGER_REMOTE"), help="default: $PEDGER_REMOTE")
    p.add_argument("--force", action="store_true", help="score again even if the remote already has scores")
    p.add_argument("--no-deadline", action="store_true", help="ignore the Kaggle 12-hour session limit")
    args = p.parse_args()

    meta = read_meta(args.store)
    items = read_eval_list(args.store, args.split, args.raw_root)
    cfg, run_dir = {}, None
    if args.run:
        run_dir = Path(args.run)
        cfg = load_config(run_dir / "config.yaml")
        ckpt = run_dir / "final.pt" if (run_dir / "final.pt").exists() else run_dir / "last.pt"
        if not (run_dir / "final.pt").exists():
            print("WARNING: run not finished, evaluating last.pt")
    else:
        ckpt = args.ckpt
    mode = args.mode or cfg.get("eval", {}).get("deploy", "ns_m")
    transform = args.transform or cfg.get("eval", {}).get("output_transform", "pedger")
    scales = (0.5, 1.0, 1.5) if args.ms else (1.0,)
    tag = f"{args.split}_{mode}{'_ms' if args.ms else ''}"
    pred_dir = Path(args.pred_dir) if args.pred_dir else (
        Path(args.out_dir) if args.out_dir else run_dir / f"pred_{tag}")
    root = run_dir if run_dir is not None else pred_dir          # where scores_<tag>.json is written
    root.mkdir(parents=True, exist_ok=True)
    out_json = root / f"scores_{tag}.json"
    per_image_json = out_json.with_name(out_json.stem + "_per_image.json")
    remote = open_remote(args.remote) if args.sync_to else None
    if args.sync_to and remote is None:
        sys.exit("--sync-to needs a remote: pass --remote or set PEDGER_REMOTE")
    mirror = Mirror(remote, args.sync_to, root)
    deadline = None if args.no_deadline else session.deadline()
    if deadline is not None:
        print(f"session time left before saving and stopping: {session.minutes_left(deadline):.0f} min", flush=True)

    # 1. already scored in an earlier session?
    if not args.predict_only and not args.force and mirror.exists(out_json):
        mirror.pull(out_json)
        mirror.pull(per_image_json)
        sc = json.load(open(out_json))
        print(f"already scored earlier: ODS {sc['ODS']:.4f}  OIS {sc['OIS']:.4f}  AP {sc['AP']:.4f}  -> {out_json}")
        return 0

    # 2. predictions: reuse local / remote ones made with the same settings, else predict
    if not args.pred_dir:
        want = {"n": len(items), "mode": mode, "transform": transform, "scales": list(scales), "ckpt": str(ckpt)}
        marker = pred_dir / PRED_MARKER

        def same(m):
            return bool(m) and all(m.get(k) == v for k, v in want.items() if k != "ckpt")

        local_ok = marker.exists() and same(json.load(open(marker)))
        if not local_ok and same(mirror.read_json(marker)):
            print("predictions already made in an earlier session: downloading them", flush=True)
            mirror.pull(pred_dir)
            local_ok = marker.exists() and same(json.load(open(marker)))
        if not local_ok:
            if session.minutes_left(deadline) < 60:
                print("not enough session time left to predict: run the notebook again to continue")
                return STOPPED_FOR_TIME
            secs = write_predictions(ckpt, items, pred_dir, mode, transform, scales, torch.device(args.device))
            print(f"predictions: {pred_dir} ({secs:.1f}s)", flush=True)
            marker.write_text(json.dumps({**want, "seconds": round(secs, 1)}, indent=1))
            mirror.push([pred_dir / f"{it['name']}.png" for it in items] + [marker], f"predictions {tag}")
    if args.predict_only:
        return 0

    # 3. score (resumes from the cache of per-image results; progress uploaded every 10 minutes)
    cache = cache_path(pred_dir, meta["max_dist"], args.thresholds)
    if not cache.exists() and mirror.exists(cache):
        mirror.pull(cache)
    if session.minutes_left(deadline) < 10:
        print("not enough session time left to score: run the notebook again to continue")
        return STOPPED_FOR_TIME
    try:
        scores, per_sample, _curve = evaluate(pred_dir, items, max_dist=meta["max_dist"], thresholds=args.thresholds,
                                              nproc=args.nproc, deadline=deadline,
                                              on_tick=lambda c: mirror.push([c], f"scoring progress {tag}"))
    except ScoringStopped as e:
        mirror.push([e.cache], f"scoring progress {tag}")
        print(f"STOPPED FOR TIME: {e.done}/{e.total} images of {tag} are scored and saved"
              f"{' to ' + remote.spec if remote else ''}. Run the notebook again (any account) to continue.")
        return STOPPED_FOR_TIME
    record = {**scores, "dataset": meta.get("dataset"), "split": args.split, "mode": mode,
              "transform": transform, "ms": args.ms, "pred_dir": str(pred_dir)}
    if run_dir is not None:
        done = json.load(open(run_dir / "done.json")) if (run_dir / "done.json").exists() else {}
        record.update({
            "run": run_dir.name, "method": cfg.get("name"), "seed": cfg.get("seed"),
            "train_split": cfg["data"]["split"], "voc": bool(cfg["data"].get("voc", False)),
            "epochs": cfg["train"]["epochs"], "samples_per_epoch": done.get("samples_per_epoch"),
            "schedule": cfg["train"].get("schedule_tag", "full"), "gpu_hours": done.get("gpu_hours"),
        })
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)
    with open(per_image_json, "w") as f:
        json.dump([{k: (float(v) if k != "name" else v) for k, v in s.items()} for s in per_sample], f)
    mirror.push([out_json, per_image_json, cache], f"scores {tag}")
    print(f"ODS {scores['ODS']:.4f}  OIS {scores['OIS']:.4f}  AP {scores['AP']:.4f}  -> {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
