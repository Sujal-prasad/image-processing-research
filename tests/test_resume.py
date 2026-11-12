"""Interrupting a run anywhere (mid-epoch) and resuming gives exactly the same model."""
import json
import shutil

import torch
from conftest import needs_pedger
from test_pipeline_smoke import tiny_cfg


def final_params(path):
    st = torch.load(path, map_location="cpu", weights_only=False)
    return {f"{k}.{n}": v for k in ("G", "G_teacher", "G_noshare", "G_teacher_noshare") for n, v in st[k].items()}


@needs_pedger
def test_mid_epoch_resume_is_exact(tiny_store):
    from pedger_kd.train_kd import Trainer
    root = tiny_store
    cfg = tiny_cfg(root, "A_agree")
    cfg["data"]["samples_per_epoch"] = 16          # 8 micro-steps per epoch, optimizer step every 2
    cfg["train"]["epochs"] = 3

    ref_dir = root / "runs" / "ref"
    assert Trainer(json.loads(json.dumps(cfg)), ref_dir, root / "data", root / "teachers", device="cpu").run()

    run_dir = root / "runs" / "interrupted"
    stops = iter([(0, 4), (1, 2), (1, 6), (2, 4)])   # four "sessions" end mid-epoch
    target = next(stops)
    sessions = 0
    while True:
        sessions += 1
        tr = Trainer(json.loads(json.dumps(cfg)), run_dir, root / "data", root / "teachers", device="cpu",
                     ckpt_minutes=0.0, should_stop=lambda e, s, t=target: (e, s) == t)
        if tr.run():
            break
        target = next(stops, (-1, -1))
        assert sessions < 10
    assert sessions == 5
    ref, got = final_params(ref_dir / "final.pt"), final_params(run_dir / "final.pt")
    for k in ref:
        assert torch.equal(ref[k], got[k]), k
    logs = [json.loads(l) for l in open(run_dir / "log.jsonl")]
    assert [r["epoch"] for r in logs] == [0, 1, 2]
    assert json.load(open(run_dir / "done.json"))["sessions"] == 5


@needs_pedger
def test_finished_run_is_not_retrained(tiny_store):
    from pedger_kd.train_kd import Trainer
    root = tiny_store
    cfg = tiny_cfg(root, "baseline")
    cfg["train"]["epochs"] = 1
    out = root / "runs" / "once"
    assert Trainer(json.loads(json.dumps(cfg)), out, root / "data", root / "teachers", device="cpu").run()
    before = (out / "final.pt").stat().st_mtime
    assert Trainer(json.loads(json.dumps(cfg)), out, root / "data", root / "teachers", device="cpu").run()
    assert (out / "final.pt").stat().st_mtime == before
    shutil.rmtree(out)
