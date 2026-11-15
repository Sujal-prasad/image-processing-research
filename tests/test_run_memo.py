"""teachers/run_memo.py drives MEMO's own scripts correctly (stand-in MEMO folder; the real model needs a GPU)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FAKE_PREDICT = """import argparse, os, shutil
from pathlib import Path
p = argparse.ArgumentParser()
for a in ["--test_folder", "--save_folder", "--config_file", "--model_path", "--base_model_path", "--guidance_scale",
          "--max_steps"]:
    p.add_argument(a)
a = p.parse_args()
assert os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") == "1"
for f in [a.config_file, a.model_path, a.base_model_path]:
    assert Path(f).is_file(), f
out = Path(a.save_folder) / "prediction"
out.mkdir(parents=True, exist_ok=True)
for img in sorted(Path(a.test_folder).glob("*.png"))[:int(os.environ.get("FAKE_LIMIT", "999"))]:
    shutil.copy(img, out / img.name)
"""


def make_memo(root):
    for rel in ["configs/discrete_BSDS_finetune/binary_lora_default.yaml",
                "pretrained_models/MEMO_BSDS_ft_late/mp_rank_00_model_states.pt",
                "pretrained_models/MEMO_synthetic_late/mp_rank_00_model_states.pt"]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x")
    (root / "download_checkpoints.py").write_text("print('checkpoints present')\n")
    (root / "edge_prediction.py").write_text(FAKE_PREDICT)
    return root


def test_run_memo(tmp_path):
    memo = make_memo(tmp_path / "memo_repo")
    imgs = tmp_path / "img"
    imgs.mkdir()
    for n in ["a", "b", "c"]:
        (imgs / f"{n}.png").write_bytes(b"png")
    cmd = [sys.executable, str(ROOT / "teachers" / "run_memo.py"), "--images", str(imgs), "--out",
           str(tmp_path / "out"), "--memo-dir", str(memo), "--no-install"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "memo: 3/3 predictions" in r.stdout
    assert sorted(p.name for p in (tmp_path / "out" / "prediction").iterdir()) == ["a.png", "b.png", "c.png"]

    r = subprocess.run([*cmd[:5], str(tmp_path / "out2"), *cmd[6:]], capture_output=True, text=True,
                       env={**os.environ, "FAKE_LIMIT": "2"})
    assert r.returncode != 0 and "fewer predictions" in r.stderr
