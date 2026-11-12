"""End to end on a tiny synthetic store: train -> checkpoint -> resume -> inference -> tables."""
import json

import pytest
import torch
from conftest import needs_pedger

from pedger_kd.config import load_config


def tiny_cfg(root, method="A_agree"):
    cfg = load_config(f"configs/{method}.yaml")
    cfg["data"].update({"store": "toy", "teacher_dir": "toy", "split": "train", "samples_per_epoch": 8})
    cfg["augment"].update({"angles_deg": [0, 90, 22.5], "scales": [1.0, 0.5]})
    cfg["train"].update({"epochs": 3, "micro_batch": 2, "accum_steps": 2, "amp": False,
                         "channels_last": False, "log_every": 1})
    return cfg


@needs_pedger
@pytest.mark.parametrize("method", ["baseline", "B_agree", "A_agree", "A_uniform"])
def test_train_resume_infer(tiny_store, method):
    from pedger_kd.inference import load_nets, predict
    from pedger_kd.train_kd import Trainer
    root = tiny_store
    cfg = tiny_cfg(root, method)
    out = root / "runs" / method
    # session 1: two of three epochs (as if a Kaggle session ended), checkpoint after each
    tr = Trainer(json.loads(json.dumps(cfg)), out, root / "data", root / "teachers", device="cpu")
    for ep in range(0, 2):
        tr.run_epoch(ep)
        for k in ("rec", "ns"):
            tr.scheds[k][0].step()
        tr.save(ep)
    # session 2: same command resumes at epoch 2 and finishes
    tr2 = Trainer(json.loads(json.dumps(cfg)), out, root / "data", root / "teachers", device="cpu")
    assert tr2.start_epoch == 2
    assert tr2.run()
    assert (out / "final.pt").exists() and (out / "done.json").exists()
    nets = load_nets(out / "final.pt", mode="ns_m")
    prob = predict(nets, torch.randn(1, 3, 40, 56), "ns_m", "pedger", scales=(0.5, 1.0, 1.5))
    assert prob.shape == (40, 56) and torch.isfinite(prob).all()
    fused = predict(load_nets(out / "final.pt", mode="fuse_m"), torch.randn(1, 3, 40, 56), "fuse_m")
    assert fused.shape == (40, 56)


def test_make_tables_groups_by_setting(tmp_path):
    import subprocess
    import sys
    base = {"dataset": "bsds", "split": "test", "train_split": "trainval", "voc": False, "ms": False,
            "schedule": "full", "mode": "ns_m", "OIS": 0.83, "AP": 0.85, "gpu_hours": 10}
    for i, (m, s, ods) in enumerate([("baseline", 0, 0.811), ("baseline", 1, 0.813), ("A_agree", 0, 0.82)]):
        json.dump({**base, "method": m, "seed": s, "ODS": ods, "run": f"r{i}"},
                  open(tmp_path / f"scores_{i}.json", "w"))
    json.dump({**base, "schedule": "short", "method": "A_agree", "seed": 0, "ODS": 0.7, "run": "x"},
              open(tmp_path / "scores_short.json", "w"))
    out = subprocess.run([sys.executable, "results/make_tables.py", "--scores", str(tmp_path / "*.json")],
                         capture_output=True, text=True, check=True).stdout
    assert out.count("### ") == 2                        # full and short never share a table
    assert "0.8120 ± 0.0014" in out                      # baseline mean +- std over 2 seeds
    assert "PEdger (BSDS only, from PEdger++ paper) †" in out
