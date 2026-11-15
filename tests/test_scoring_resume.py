"""Resumable scoring: per-image cache, deadline stop + resume, progress sync to the remote, gate check.

A stand-in pyEdgeEval package (logs each per-image evaluation, can sleep) replaces the real benchmark.
"""
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import pytest
from conftest import ROOT

from pedger_kd import evaluation, session

STUB = {
    "pyEdgeEval/__init__.py": "",
    "pyEdgeEval/_lib/__init__.py": "def correspond_pixels(*a, **k): pass\ndef nms(*a, **k): pass\n",
    "pyEdgeEval/common/__init__.py": "",
    "pyEdgeEval/common/utils.py": "import numpy as np\ndef check_thresholds(t):\n"
                                  "    return np.linspace(1/(t+1), 1-1/(t+1), t) if isinstance(t, int) else np.array(t)\n",
    "pyEdgeEval/common/binary_label/evaluate_boundaries.py":
        "import os, time\nimport numpy as np\n"
        "def evaluate_boundaries_threshold_multiple_gts(thresholds, pred, gts, **kw):\n"
        "    if os.environ.get('STUB_LOG'):\n"
        "        with open(os.environ['STUB_LOG'], 'a') as f: f.write('x\\n')\n"
        "    time.sleep(float(os.environ.get('STUB_SLEEP', '0')))\n"
        "    z = np.full(len(thresholds), float(pred.max() + len(gts)))\n    return z, z, z, z\n",
    "pyEdgeEval/common/binary_label/__init__.py":
        "import numpy as np\n"
        "def calculate_metrics(eval_single, thresholds, samples, nproc=1):\n"
        "    res = [eval_single(s) for s in samples]\n"
        "    per = [dict(name=s['name'], f1=np.float64(r[0][0])) for s, r in zip(samples, res)]\n"
        "    tot = float(sum(r[0][0] for r in res))\n"
        "    overall = {k: np.float64(tot) for k in ('ODS_threshold', 'ODS_recall', 'ODS_precision', 'ODS_f1',\n"
        "               'OIS_f1', 'AP')}\n"
        "    return per, [dict(f1=np.float64(tot))], overall\n",
}


def make_stub(tmp_path):
    stub = tmp_path / "stub"
    for rel, text in STUB.items():
        (stub / rel).parent.mkdir(parents=True, exist_ok=True)
        (stub / rel).write_text(text)
    return stub


def make_data(tmp_path, n=3):
    preds, raw = tmp_path / "pred", tmp_path / "raw"
    preds.mkdir()
    (raw / "gt").mkdir(parents=True)
    items = []
    for i in range(n):
        img = np.zeros((20, 30), np.uint8)
        img[10, 5:25] = 255
        cv2.imwrite(str(preds / f"im{i}.png"), img)
        cv2.imwrite(str(raw / "gt" / f"im{i}.png"), img)
        items.append({"name": f"im{i}", "gt": str(raw / "gt" / f"im{i}.png")})
    return preds, raw, items


def calls(log):
    return len(log.read_text().splitlines()) if log.exists() else 0


@pytest.fixture
def stub_env(tmp_path, monkeypatch):
    stub = make_stub(tmp_path)
    log = tmp_path / "calls.log"
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(stub), str(ROOT), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("STUB_LOG", str(log))
    monkeypatch.setattr(evaluation, "_SCORER", None)          # "pyEdgeEval works in this Python"
    return stub, log


@pytest.fixture
def stub_in_process(stub_env, monkeypatch):
    stub, log = stub_env
    monkeypatch.syspath_prepend(str(stub))
    yield stub, log
    for m in [m for m in sys.modules if m == "pyEdgeEval" or m.startswith("pyEdgeEval.")]:
        del sys.modules[m]


def test_cache_resumes_and_redoes_changed_predictions(tmp_path, stub_env):
    _, log = stub_env
    preds, _, items = make_data(tmp_path)
    s1, per1, _ = evaluation.evaluate(preds, items, nproc=1)
    assert calls(log) == 3 and s1["ODS"] == 6.0 and [p["name"] for p in per1] == ["im0", "im1", "im2"]
    cache = evaluation.cache_path(preds)
    assert len(cache.read_text().splitlines()) == 3
    s2, _, _ = evaluation.evaluate(preds, items, nproc=1)
    assert calls(log) == 3 and s2 == s1                               # everything came from the cache
    with open(cache, "a") as f:                                       # a line cut off by a killed session
        f.write('{"name": "im1", "md5"')
    img = np.zeros((20, 30), np.uint8)
    img[3, 3:9] = 128                                                 # prediction im2 changed -> scored again
    cv2.imwrite(str(preds / "im2.png"), img)
    s3, per3, _ = evaluation.evaluate(preds, items, nproc=1)
    assert calls(log) == 4
    assert per3[2]["f1"] == pytest.approx(128 / 255 + 1) and s3["ODS"] == pytest.approx(4 + 128 / 255 + 1)


def _pools_work():
    try:
        import multiprocessing
        multiprocessing.Lock()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _pools_work(), reason="process pools are blocked in this environment")
def test_process_pool_gives_same_scores(tmp_path, stub_env):
    _, log = stub_env
    preds, _, items = make_data(tmp_path, n=5)
    s, per, _ = evaluation.evaluate(preds, items, nproc=3)
    assert calls(log) == 5 and s["ODS"] == 10.0 and [p["name"] for p in per] == [f"im{i}" for i in range(5)]


def test_deadline_stops_then_resumes(tmp_path, stub_in_process, monkeypatch):
    _, log = stub_in_process
    preds, _, items = make_data(tmp_path)
    monkeypatch.setenv("STUB_SLEEP", "0.5")
    res = evaluation.score_here(str(preds), items, nproc=1, deadline=time.time() + 0.7)
    assert res["stopped"] and 1 <= res["done"] < 3 and res["total"] == 3 and calls(log) == res["done"]
    res = evaluation.score_here(str(preds), items, nproc=1, deadline=None)
    assert res["scores"]["ODS"] == 6.0 and calls(log) == 3          # only the missing images were scored


def test_deadline_already_passed_raises_scoring_stopped(tmp_path, stub_env):
    _, log = stub_env
    preds, _, items = make_data(tmp_path)
    with pytest.raises(evaluation.ScoringStopped) as e:
        evaluation.evaluate(preds, items, nproc=1, deadline=time.time() - 1)
    assert e.value.done == 0 and e.value.total == 3 and calls(log) == 0


def test_run_bsds_eval_syncs_and_resumes_from_remote(tmp_path, stub_env):
    _, log = stub_env
    preds, raw, items = make_data(tmp_path)
    store = tmp_path / "store"
    (store / "eval_lists").mkdir(parents=True)
    (store / "meta.json").write_text(json.dumps({"dataset": "toy", "max_dist": 0.0075}))
    (store / "eval_lists" / "test.json").write_text(json.dumps(
        [{"name": it["name"], "image": f"gt/{it['name']}.png", "gt": f"gt/{it['name']}.png"} for it in items]))
    remote = tmp_path / "remote"

    def run(pred_dir):
        r = subprocess.run([sys.executable, str(ROOT / "eval" / "run_bsds_eval.py"), "--pred-dir", str(pred_dir),
                            "--store", str(store), "--split", "test", "--raw-root", str(raw), "--nproc", "1",
                            "--remote", f"local:{remote}", "--sync-to", "results/toy", "--device", "cpu"],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout

    out = run(preds)
    assert calls(log) == 3 and "ODS 6.0000" in out
    rfiles = sorted(p.name for p in (remote / "results" / "toy").iterdir())
    assert "scores_test_ns_m.json" in rfiles and evaluation.cache_path(preds).name in rfiles
    # a new session: same predictions, nothing local -> the scores come from the remote, nothing is re-scored
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    for it in items:
        (fresh / f"{it['name']}.png").write_bytes((preds / f"{it['name']}.png").read_bytes())
    assert "already scored earlier" in run(fresh) and calls(log) == 3
    # scores missing on the remote but the progress cache is there -> resumes from it, nothing re-scored
    (remote / "results" / "toy" / "scores_test_ns_m.json").unlink()
    fresh2 = tmp_path / "fresh2"
    fresh2.mkdir()
    for it in items:
        (fresh2 / f"{it['name']}.png").write_bytes((preds / f"{it['name']}.png").read_bytes())
    assert "ODS 6.0000" in run(fresh2) and calls(log) == 3


def test_session_deadline(monkeypatch):
    monkeypatch.setenv("PEDGER_SESSION_START", "1000")
    assert session.deadline(reserve_minutes=45) == 1000 + 12 * 3600 - 45 * 60
    monkeypatch.delenv("PEDGER_SESSION_START")
    if not session.on_kaggle():
        assert session.deadline() is None and session.minutes_left(None) == float("inf")


def test_gate_tolerance():
    sys.path.insert(0, str(ROOT / "eval"))
    import released_gate as g
    assert g.matches(0.8215, 0.8390) == ["BSDS+VOC"]
    assert g.matches(0.8125, 0.8355) == ["BSDS only"]
    assert g.matches(0.8180, 0.8370) == []
    assert [s[0] for s in g.sets(["ns_m", "rec_m"], True)] == ["ns_m", "rec_m", "ns_m_ms", "rec_m_ms"]
