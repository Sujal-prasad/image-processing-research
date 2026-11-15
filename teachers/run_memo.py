"""Run MEMO (teacher 3) on a folder of exported images; writes <out>/prediction/<name>.png.

    python teachers/run_memo.py --images work/img --out work/memo

Installs only what MEMO's prediction code imports (its requirements.txt is unpinned, would replace Kaggle's torch,
lists packages inference never uses and omits peft), keeping the installed torch/torchvision, then downloads its
official checkpoints (skipped if present) and runs its edge_prediction.py.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPS = ["pytorch_lightning", "omegaconf", "diffusers", "einops", "peft"]
CONFIG = "configs/discrete_BSDS_finetune/binary_lora_default.yaml"
MODEL = "pretrained_models/MEMO_BSDS_ft_late/mp_rank_00_model_states.pt"
BASE = "pretrained_models/MEMO_synthetic_late/mp_rank_00_model_states.pt"


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--memo-dir", default=str(ROOT / "third_party" / "MEMO_Edge_Detection"))
    p.add_argument("--no-install", action="store_true")
    args = p.parse_args()
    images, out = Path(args.images).resolve(), Path(args.out).resolve()
    if not args.no_install:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as pins:
            pins.write(f"torch=={version('torch')}\ntorchvision=={version('torchvision')}\n")
        run([sys.executable, "-m", "pip", "install", "-q", *DEPS, "-c", pins.name])
    run([sys.executable, "download_checkpoints.py", "--model", "bsds-late", "--model", "synthetic-late"],
        cwd=args.memo_dir)
    # The checkpoints also pickle omegaconf configs, which torch >= 2.6 refuses by default (weights_only=True).
    # They come from MEMO's official Hugging Face page via its own download script, so allow it for this step only.
    env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    run([sys.executable, "edge_prediction.py", "--test_folder", images, "--save_folder", out, "--config_file", CONFIG,
         "--model_path", MODEL, "--base_model_path", BASE, "--guidance_scale", "1.4", "--max_steps", "20"],
        cwd=args.memo_dir, env=env)
    n_in, n_out = len(list(images.glob("*.png"))), len(list((out / "prediction").glob("*.png")))
    print(f"memo: {n_out}/{n_in} predictions in {out / 'prediction'}", flush=True)
    if n_out < n_in:
        raise SystemExit("MEMO wrote fewer predictions than there are images; see its output above")


if __name__ == "__main__":
    main()
