"""Predict (optional) and score ODS / OIS / AP on a split (tasks 1 and 10).

Score a trained run (writes <run>/scores_<split>_<mode>[_ms].json for make_tables.py):
    python eval/run_bsds_eval.py --run runs/A_agree_s0 --store data_store/bsds --split test \
        --raw-root BSR/BSDS500/data
Step 1, the released PEdger checkpoint (try the modes until one reproduces the paper):
    python eval/run_bsds_eval.py --ckpt third_party/PEdger/checkpoint_PEdger.pth --mode ns_m \
        --store data_store/bsds --split test --raw-root BSR/BSDS500/data --out-dir outputs/released_ns_m
Score an existing folder of PNG predictions (e.g. a teacher):
    python eval/run_bsds_eval.py --pred-dir work/test_sauge --store data_store/bsds --split test --raw-root ...
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pedger_kd.augment import normalize_image_u8  # noqa: E402
from pedger_kd.config import load_config  # noqa: E402
from pedger_kd.data import read_meta  # noqa: E402
from pedger_kd.evaluation import evaluate, read_eval_list  # noqa: E402
from pedger_kd.inference import load_nets, predict  # noqa: E402


def write_predictions(ckpt, items, out_dir, mode, transform, scales, device):
    nets = load_nets(ckpt, device=device, mode=mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for it in tqdm(items, desc=f"predict {mode}"):
        img = cv2.cvtColor(cv2.imread(it["image"]), cv2.COLOR_BGR2RGB)
        prob = predict(nets, normalize_image_u8(img, device), mode, transform, scales)
        cv2.imwrite(str(out_dir / f"{it['name']}.png"), (prob.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy())
    return time.time() - t0


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
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--thresholds", type=int, default=99)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--predict-only", action="store_true")
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

    if args.pred_dir:
        pred_dir = Path(args.pred_dir)
    else:
        pred_dir = Path(args.out_dir) if args.out_dir else (run_dir / f"pred_{tag}")
        secs = write_predictions(ckpt, items, pred_dir, mode, transform, scales, torch.device(args.device))
        print(f"predictions: {pred_dir} ({secs:.1f}s)")
    if args.predict_only:
        return

    scores, per_sample, _curve = evaluate(pred_dir, items, max_dist=meta["max_dist"],
                                          thresholds=args.thresholds, nproc=args.nproc)
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
    out_json = (run_dir / f"scores_{tag}.json") if run_dir else (pred_dir / f"scores_{tag}.json")
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)
    with open(out_json.with_name(out_json.stem + "_per_image.json"), "w") as f:
        json.dump([{k: (float(v) if k != "name" else v) for k, v in s.items()} for s in per_sample], f)
    print(f"ODS {scores['ODS']:.4f}  OIS {scores['OIS']:.4f}  AP {scores['AP']:.4f}  -> {out_json}")


if __name__ == "__main__":
    main()
