"""The job queue across sessions and accounts, with a local folder as the remote."""
import json
import shutil
import time

import yaml
from conftest import needs_pedger

from pedger_kd.sync import LocalRemote, iso

TINY = ["data.store=toy", "data.teacher_dir=toy", "data.split=train", "data.samples_per_epoch=16",
        "augment.angles_deg=[0, 90, 22.5]", "augment.scales=[1.0, 0.5]", "train.epochs=2",
        "train.micro_batch=2", "train.accum_steps=2", "train.amp=false", "train.channels_last=false",
        "train.log_every=100"]


def make_remote(tiny_store, tmp_path):
    remote = LocalRemote(tmp_path / "remote")
    shutil.copytree(tiny_store / "data" / "toy", remote.root / "data" / "toy")
    shutil.copytree(tiny_store / "teachers" / "toy", remote.root / "teachers" / "toy")
    return remote


def write_queue(path, entries):
    path.write_text(yaml.safe_dump({"jobs": entries}))
    return path


def worker(remote, ws, wid, **kw):
    from pedger_kd.jobs import Worker
    w = Worker(remote, ws, wid, deadline=time.time() + 3600, settle_seconds=0, ckpt_minutes=0.0, device="cpu", **kw)
    remote.download("data/toy", ws)
    remote.download("teachers/toy", ws)
    return w


@needs_pedger
def test_queue_moves_across_accounts(tiny_store, tmp_path):
    from pedger_kd.jobs import load_queue
    remote = make_remote(tiny_store, tmp_path)
    q = load_queue(write_queue(tmp_path / "q.yaml", [
        {"config": "configs/A_agree.yaml", "seeds": [0], "set": TINY},
        {"config": "configs/baseline.yaml", "seeds": [0], "set": TINY},
        {"config": "configs/A_uniform.yaml", "seeds": [0], "set": TINY + ["data.teacher=missing_teacher"]},
    ]))
    # account 1: session ends in the middle of epoch 1 of the first job
    w1 = worker(remote, tmp_path / "ws_account1", "account1")
    w1.trainer_kwargs = {"should_stop": lambda e, s: (e, s) == (1, 4)}
    assert w1.run(q) == {"A_agree_s0": "stopped"}
    p = remote.read_json("runs/A_agree_s0/progress.json")
    assert p["status"] == "stopped" and p["epoch"] == 1 and p["step"] == 4
    assert remote.exists("runs/A_agree_s0/last.pt") and not remote.exists("runs/A_agree_s0/final.pt")

    # account 2 (fresh machine, empty workspace) continues the same job, then the rest of the queue
    w2 = worker(remote, tmp_path / "ws_account2", "account2")
    summary = w2.run(q)
    assert summary == {"A_agree_s0": "done", "baseline_s0": "done", "A_uniform_s0": "failed"}
    for run in ("A_agree_s0", "baseline_s0"):
        assert remote.read_json(f"runs/{run}/progress.json")["status"] == "done"
        assert remote.exists(f"runs/{run}/final.pt") and remote.exists(f"runs/{run}/done.json")
    assert json.loads((remote.root / "runs/A_agree_s0/done.json").read_text())["sessions"] == 2
    failed = remote.read_json("runs/A_uniform_s0/progress.json")
    assert failed["status"] == "failed" and "missing_teacher" in failed["error"]

    # a later session skips finished and failed jobs
    w3 = worker(remote, tmp_path / "ws_account3", "account3")
    assert w3.run(q) == {"A_agree_s0": "done", "baseline_s0": "done", "A_uniform_s0": "failed-before"}


@needs_pedger
def test_lock_respects_live_workers_and_takes_over_stale_ones(tiny_store, tmp_path):
    remote = make_remote(tiny_store, tmp_path)
    prog = tmp_path / "p.json"
    prog.write_text(json.dumps({"status": "running", "worker": "other-account", "heartbeat": iso()}))
    remote.upload({"runs/x_s0/progress.json": str(prog)})
    w = worker(remote, tmp_path / "ws", "me", stale_minutes=75)
    assert w.claim("x_s0") == "busy"
    prog.write_text(json.dumps({"status": "running", "worker": "other-account",
                                "heartbeat": "2020-01-01T00:00:00+00:00"}))   # crashed long ago
    remote.upload({"runs/x_s0/progress.json": str(prog)})
    assert w.claim("x_s0") == "claimed"
    assert remote.read_json("runs/x_s0/progress.json")["worker"] == "me"


@needs_pedger
def test_cli_exactly_as_the_notebook_runs_it(tiny_store, tmp_path):
    """python -m pedger_kd.jobs ... (pulls data from the remote, spawns a worker, trains, uploads)."""
    import os
    import subprocess
    import sys
    from conftest import ROOT
    remote = make_remote(tiny_store, tmp_path)
    q = write_queue(tmp_path / "q.yaml", [{"config": "configs/baseline.yaml", "seeds": [3], "set": TINY}])
    env = {**os.environ, "PEDGER_REMOTE": remote.spec}
    cmd = [sys.executable, "-m", "pedger_kd.jobs", "--queue", str(q), "--ws", str(tmp_path / "ws"), "--hours", "1",
           "--session-start", str(time.time()), "--gpus", "none", "--label", "test-account", "--settle-seconds", "0"]
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "pulled data/toy" in out.stdout and '"baseline_s3": "done"' in out.stdout
    p = remote.read_json("runs/baseline_s3/progress.json")
    assert p["status"] == "done" and p["worker"].startswith("test-account")
    st = subprocess.run([sys.executable, "-m", "pedger_kd.sync", "status"], cwd=ROOT, env=env,
                        capture_output=True, text=True, check=True).stdout
    assert "baseline_s3" in st and "done" in st


def test_async_pusher_uploads_latest_and_survives_overwrites(tmp_path):
    from pedger_kd.sync import AsyncPusher
    remote = LocalRemote(tmp_path / "remote")
    f = tmp_path / "last.pt"
    pusher = AsyncPusher(remote, tmp_path / "staging")
    for i in range(5):
        f.write_text(f"checkpoint {i}")
        pusher.request({"runs/r/last.pt": str(f)}, f"ckpt {i}")
    assert pusher.flush(timeout=30)
    assert (remote.root / "runs/r/last.pt").read_text() == "checkpoint 4"
    assert not any((tmp_path / "staging").iterdir())
