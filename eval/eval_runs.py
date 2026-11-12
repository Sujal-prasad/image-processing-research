"""Score every finished run on the remote (restartable; CPU session).

For each runs/<run>/done.json on the remote without a scores file for the chosen
setting: pull the run, predict, score, push scores (+ predictions for the test
split, used by the agreement-bin analysis). Already-scored runs are skipped, so the
notebook can be re-run from any account at any time.

Split: runs trained on `train` (dev/ablations) are scored on val, runs trained on
`trainval` on test.

    python eval/eval_runs.py --remote hf:you/pedger-store --ws /kaggle/working/ws \
        --raw-root bsds=/kaggle/working/BSR/BSDS500/data [--ms] [--runs "A_agree_*"]
"""
import argparse
import fnmatch
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pedger_kd.config import load_config  # noqa: E402
from pedger_kd.sync import open_remote  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--remote", required=True)
    p.add_argument("--ws", required=True)
    p.add_argument("--raw-root", action="append", required=True, help="dataset=path, e.g. bsds=/.../BSDS500/data")
    p.add_argument("--runs", default="*", help="glob on run names")
    p.add_argument("--ms", action="store_true")
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    remote = open_remote(args.remote)
    ws = Path(args.ws)
    raw = dict(r.split("=", 1) for r in args.raw_root)
    files = remote.list("runs")
    done = sorted({f.split("/")[1] for f in files if f.endswith("/done.json")})
    done = [r for r in done if fnmatch.fnmatch(r, args.runs)]
    print(f"{len(done)} finished runs on {remote.spec}")
    for run in done:
        meta_cfg = remote.download(f"runs/{run}/config.yaml", ws)
        cfg = load_config(ws / "runs" / run / "config.yaml") if meta_cfg else None
        if cfg is None:
            continue
        store = cfg["data"]["store"]
        dataset_meta_ok = remote.download(f"data/{store}/meta.json", ws) and remote.download(f"data/{store}/eval_lists", ws)
        dataset = json.load(open(ws / "data" / store / "meta.json"))["dataset"] if dataset_meta_ok else None
        root_key = dataset if dataset in raw else next((k for k in raw if store.startswith(k)), None)
        if root_key is None:
            print(f"skip {run}: no --raw-root for dataset {dataset!r} (store {store})")
            continue
        split = "val" if cfg["data"]["split"] == "train" else "test"
        mode = cfg.get("eval", {}).get("deploy", "ns_m")
        tag = f"{split}_{mode}{'_ms' if args.ms else ''}"
        if any(f == f"runs/{run}/scores_{tag}.json" for f in files):
            print(f"skip {run}: scores_{tag} already on remote")
            continue
        for f in ("final.pt", "done.json"):
            remote.download(f"runs/{run}/{f}", ws)
        cmd = [sys.executable, str(ROOT / "eval" / "run_bsds_eval.py"), "--run", str(ws / "runs" / run),
               "--store", str(ws / "data" / store), "--split", split, "--raw-root", raw[root_key],
               "--nproc", str(args.nproc), "--device", args.device] + (["--ms"] if args.ms else [])
        print(f"=== {run}: {split}{' MS' if args.ms else ''}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            print(f"FAILED {run}")
            continue
        run_dir = ws / "runs" / run
        up = {f"runs/{run}/scores_{tag}.json": str(run_dir / f"scores_{tag}.json"),
              f"runs/{run}/scores_{tag}_per_image.json": str(run_dir / f"scores_{tag}_per_image.json")}
        if split == "test":
            for png in (run_dir / f"pred_{tag}").glob("*.png"):
                up[f"runs/{run}/pred_{tag}/{png.name}"] = str(png)
        remote.upload(up, f"scores {run} {tag}")
    # pull every score (also those computed in earlier sessions) for the tables
    for f in remote.list("runs"):
        if "/scores_" in f and f.endswith(".json") and not (ws / f).exists():
            remote.download(f, ws)


if __name__ == "__main__":
    main()
