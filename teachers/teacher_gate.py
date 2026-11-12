"""Teacher gate (plan 8.4 / task 4): keep a teacher on a dataset only if it beats
PEdger on that dataset's validation split.

BSDS500: SAUGE, MuGE and MEMO were trained on BSDS train+val, so the BSDS val
split is in-sample for them and a gate there is meaningless. Use their published
test scores for BSDS and run this gate only on NYUDv2 / MultiCue.

    python teachers/teacher_gate.py --store data_store/nyud --split val --raw-root NYUD \
        --teacher sauge=work/val_sauge --teacher muge=work/val_muge --teacher memo=work/val_memo/prediction \
        --teacher ensemble=work/val_ensemble --baseline-scores runs/baseline_nyud_s0/scores_val.json \
        --out teacher_cache/nyud/gate.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pedger_kd.data import read_meta  # noqa: E402
from pedger_kd.evaluation import evaluate, read_eval_list  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--raw-root", required=True)
    p.add_argument("--teacher", action="append", required=True, help="name=png_dir")
    p.add_argument("--baseline-scores", required=True, help="scores json of PEdger on the same split")
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    meta = read_meta(args.store)
    if meta.get("dataset") == "bsds":
        print("WARNING: BSDS val is in-sample for BSDS-trained teachers; this gate is not meaningful on BSDS.")
    items = read_eval_list(args.store, args.split, args.raw_root)
    base = json.load(open(args.baseline_scores))["ODS"]
    result = {"dataset": meta.get("dataset"), "split": args.split, "pedger_ODS": base, "teachers": {}}
    for spec in args.teacher:
        name, pred_dir = spec.split("=", 1)
        scores, _, _ = evaluate(pred_dir, items, max_dist=meta["max_dist"], nproc=args.nproc)
        keep = scores["ODS"] > base
        result["teachers"][name] = {**scores, "keep": keep}
        print(f"{name:12s} ODS {scores['ODS']:.4f} OIS {scores['OIS']:.4f} AP {scores['AP']:.4f} "
              f"vs PEdger {base:.4f} -> {'KEEP' if keep else 'DROP'}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
