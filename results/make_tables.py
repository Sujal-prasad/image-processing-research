"""Build the paper tables from scores_*.json files.

Rows are grouped by method (config name) and reported as mean +- std over seeds.
A table is built per SETTING (dataset, split, train split, +VOC or not, SS/MS,
schedule); runs from different settings are never placed in the same table, and
published numbers (results/published.yaml) are only shown in the table whose
setting they match, marked with a dagger.

    python results/make_tables.py --scores "runs/*/scores_test_*.json" --out results/tables.md
"""
import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SETTING_KEYS = ("dataset", "split", "train_split", "voc", "ms", "schedule", "mode")
ORDER = ["baseline", "B_uniform", "B_agree", "A_uniform", "A_agree"]


def setting_of(r):
    return tuple((k, r.get(k)) for k in SETTING_KEYS)


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    m = statistics.mean(vals)
    if len(vals) == 1:
        return f"{m:.4f}"
    return f"{m:.4f} ± {statistics.stdev(vals):.4f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", nargs="+", required=True, help="glob(s) of scores json files from runs")
    p.add_argument("--published", default=str(HERE / "published.yaml"))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    files = sorted({f for pat in args.scores for f in glob.glob(pat) if not f.endswith("_per_image.json")})
    recs = [json.load(open(f)) for f in files]
    recs = [r for r in recs if r.get("method")]
    published = yaml.safe_load(open(args.published)) if Path(args.published).exists() else []

    groups = defaultdict(lambda: defaultdict(list))
    for r in recs:
        groups[setting_of(r)][r["method"]].append(r)
    lines = []
    for setting, methods in groups.items():
        s = dict(setting)
        title = (f"{s['dataset']} {s['split']} | trained on {s['train_split']}{' + VOC' if s['voc'] else ''} | "
                 f"{'MS' if s['ms'] else 'SS'} | schedule {s['schedule']} | output {s['mode']}")
        lines += [f"### {title}", "", "| Method | n seeds | ODS | OIS | AP | train GPU-h |",
                  "|---|---|---|---|---|---|"]
        for pub in published:
            if all(pub.get(k) == s.get(k) for k in ("dataset", "split", "voc", "ms")):
                lines.append(f"| {pub['method']} † | - | {pub.get('ODS', '-')} | {pub.get('OIS', '-')} | "
                             f"{pub.get('AP', '-')} | - |")
        for m in sorted(methods, key=lambda x: (ORDER.index(x) if x in ORDER else len(ORDER), x)):
            rs = methods[m]
            seeds = sorted({r.get("seed") for r in rs})
            if len(seeds) != len(rs):
                raise ValueError(f"duplicate seeds for {m} in setting {title}: {[r.get('run') for r in rs]}")
            lines.append(f"| {m} | {len(rs)} | {fmt([r['ODS'] for r in rs])} | {fmt([r['OIS'] for r in rs])} | "
                         f"{fmt([r['AP'] for r in rs])} | {fmt([r.get('gpu_hours') for r in rs])} |")
        lines += ["", "† published number, same setting; not retrained by us.", ""]
    text = "\n".join(lines) if lines else "no scores found"
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
