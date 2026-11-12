from pathlib import Path

import pytest

from pedger_kd.config import load_config

CONFIGS = sorted(p for p in Path("configs").rglob("*.yaml") if not p.name.startswith("_") and p.name != "base.yaml")


@pytest.mark.parametrize("path", CONFIGS, ids=[str(p) for p in CONFIGS])
def test_every_config_is_complete(path):
    cfg = load_config(path)
    assert cfg["name"] and cfg["name"] != "base"
    assert cfg["train"]["micro_batch"] * cfg["train"]["accum_steps"] == 16
    if cfg["kd"]["enabled"] or cfg["method"]["target"] == "agreement":
        assert cfg["data"]["teacher"], "teacher-based method without a teacher"
    if cfg["method"]["alpha_mode"] == "agreement" and cfg["method"]["target"] == "agreement":
        assert cfg["data"]["store"] != "nyud", "NYUDv2 has no agreement map"


def test_overrides_and_inheritance():
    cfg = load_config("configs/ablations/gamma_conf.yaml", ["train.epochs=5", "method.alpha_min=0.2"])
    assert cfg["method"]["gamma"] == "confidence" and cfg["method"]["target"] == "agreement"
    assert cfg["data"]["split"] == "train" and cfg["train"]["schedule_tag"] == "short"
    assert cfg["train"]["epochs"] == 5 and cfg["method"]["alpha_min"] == 0.2


def test_main_rows_differ_only_where_intended():
    a, u = load_config("configs/A_agree.yaml"), load_config("configs/A_uniform.yaml")
    assert a["data"] == u["data"] and a["train"] == u["train"] and a["kd"] == u["kd"]
    assert a["method"]["alpha_mode"] == "agreement" and u["method"]["alpha_mode"] == "eta"
