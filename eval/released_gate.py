"""Step 1 gate: score the released PEdger checkpoint and compare it with the published numbers.

One "set" = one inference mode, single-scale (SS) or multi-scale (MS). All SS sets come first (the
gate only needs those), MS sets follow with --ms. Scoring one set of 200 test images takes ~4 hours on
Kaggle's CPUs, so this is resumable: every set, and the progress of the unfinished one, is kept on the
remote under results/released/<set>/. Running it again (any session, any account) skips finished sets
and continues the unfinished one. It stops before the Kaggle 12-hour limit.

    python eval/released_gate.py --store ws/data/bsds --raw-root BSR/BSDS500/data [--modes ns_m rec_m fuse_m] [--ms]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pedger_kd.sync import open_remote  # noqa: E402

PUBLISHED = {"BSDS+VOC": (0.821, 0.840), "BSDS only": (0.813, 0.834)}   # PEdger paper, ODS / OIS
TOL = 0.002
STOPPED_FOR_TIME = 3


def matches(ods, ois, tol=TOL):
    """Names of the published settings this ODS/OIS reproduces (both within tol)."""
    return [k for k, (o, i) in PUBLISHED.items() if abs(ods - o) <= tol + 1e-9 and abs(ois - i) <= tol + 1e-9]


def sets(modes, ms):
    """[(name, mode, multi_scale)]: every single-scale set first, then the multi-scale ones."""
    return [(m, m, False) for m in modes] + ([(f"{m}_ms", m, True) for m in modes] if ms else [])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--ckpt", default=str(ROOT / "third_party" / "PEdger" / "checkpoint_PEdger.pth"))
    p.add_argument("--modes", nargs="+", default=["ns_m", "rec_m", "fuse_m"])
    p.add_argument("--ms", action="store_true", help="also the multi-scale sets (after all single-scale ones)")
    p.add_argument("--out", default="/kaggle/working/released" if Path("/kaggle").exists() else "outputs/released")
    p.add_argument("--remote", default=os.environ.get("PEDGER_REMOTE"))
    p.add_argument("--prefix", default="results/released")
    p.add_argument("--device", default="cpu")
    p.add_argument("--nproc", type=int, default=0)
    args = p.parse_args()
    remote = open_remote(args.remote) if args.remote else None
    out = Path(args.out)

    stopped = False
    for name, mode, ms in sets(args.modes, args.ms):
        cmd =[sys.executable, str(ROOT / "eval" / "run_bsds_eval.py"), "--ckpt", args.ckpt, "--mode", mode,
               "--store", args.store, "--split", args.split, "--raw-root", args.raw_root,
               "--out-dir", str(out / name), "--device", args.device, "--nproc", str(args.nproc)]
        cmd += ["--ms"] if ms else []
        cmd += ["--remote", args.remote, "--sync-to", f"{args.prefix}/{name}"] if remote else []
        print(f"\n=== released checkpoint, {name} ({'multi' if ms else 'single'}-scale)", flush=True)
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        if rc == STOPPED_FOR_TIME:
            stopped = True
            break
        if rc != 0:
            print(f"FAILED {name} (error above); continuing with the next set")

    # summary of every set scored so far, in this or earlier sessions
    rows, pending, deploy = [], [], None
    for name, _mode, ms in sets(args.modes, args.ms):
        f = out / name / f"scores_{args.split}_{name}.json"
        sc = json.load(open(f)) if f.exists() else (
            remote.read_json(f"{args.prefix}/{name}/scores_{args.split}_{name}.json") if remote else None)
        if sc is None:
            pending.append((name, ms))
            continue
        m = matches(sc["ODS"], sc["OIS"])
        if m and deploy is None and not ms:
            deploy = name
        rows.append(f"{name:10s} ODS {sc['ODS']:.4f}  OIS {sc['OIS']:.4f}  AP {sc['AP']:.4f}   "
                    f"{'reproduces ' + ' / '.join(m) if m else '-'}")
    lines = ["released PEdger checkpoint, BSDS500 test (200 images)"] + rows + [
        "published (ODS/OIS): " + "   ".join(f"{k} .{round(o * 1000)}/.{round(i * 1000)}" for k, (o, i) in PUBLISHED.items())
        + f"   (gate: both within +/-{TOL})"]
    if deploy:
        lines.append(f"GATE PASSED: mode {deploy} reproduces the paper -> eval.deploy: {deploy} in configs/base.yaml")
    elif not [n for n, ms in pending if not ms]:
        lines.append("GATE NOT PASSED: no single-scale mode reproduces the paper (check the evaluation settings)")
    pending = [n for n, _ in pending]
    if pending:
        lines.append(f"not scored yet: {', '.join(pending)}"
                     + ("  -> run the notebook again to continue" if stopped else ""))
    text = "\n".join(lines)
    print("\n" + text)
    out.mkdir(parents=True, exist_ok=True)
    (out / "released_scores.txt").write_text(text + "\n")
    summary = {"rows": rows, "pending": pending, "deploy": deploy, "published": PUBLISHED, "tolerance": TOL}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if remote:
        remote.upload({f"{args.prefix}/released_scores.txt": str(out / "released_scores.txt"),
                       f"{args.prefix}/summary.json": str(out / "summary.json")}, "released checkpoint summary")


if __name__ == "__main__":
    main()
