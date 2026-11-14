"""Scoring through a separate interpreter (the Python 3.8 fallback of tools/install_pyedgeeval.py).

A stand-in pyEdgeEval package is put on the child's PYTHONPATH, so the full path is exercised:
evaluate() -> score_worker in another process -> evaluate() there -> JSON back.
"""
import ast
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from conftest import ROOT

from pedger_kd import evaluation

STUB = {
    "pyEdgeEval/__init__.py": "",
    "pyEdgeEval/_lib/__init__.py": "def correspond_pixels(*a, **k): pass\ndef nms(*a, **k): pass\n",
    "pyEdgeEval/common/__init__.py": "",
    "pyEdgeEval/common/utils.py": "import numpy as np\ndef check_thresholds(t):\n"
                                  "    return np.linspace(1/(t+1), 1-1/(t+1), t) if isinstance(t, int) else np.array(t)\n",
    "pyEdgeEval/common/binary_label/evaluate_boundaries.py":
        "import numpy as np\ndef evaluate_boundaries_threshold_multiple_gts(thresholds, pred, gts, **kw):\n"
        "    z = np.full(len(thresholds), float(pred.max() + len(gts)))\n    return z, z, z, z\n",
    "pyEdgeEval/common/binary_label/__init__.py":
        "import os\nimport numpy as np\n"
        "def calculate_metrics(eval_single, thresholds, samples, nproc=1):\n"
        "    res = [eval_single(s) for s in samples]\n"
        "    per = [dict(name=s['name'], threshold=np.float64(0.5), f1=np.float64(r[0][0])) for s, r in zip(samples, res)]\n"
        "    overall = {k: np.float64(v) for k, v in dict(ODS_threshold=0.5, ODS_recall=0.8, ODS_precision=0.9,\n"
        "               ODS_f1=0.811, OIS_f1=0.83, AP=0.85, AUC=0.8, OIS_recall=0.8, OIS_precision=0.9).items()}\n"
        "    overall['pid'] = os.getpid()\n"
        "    return per, [dict(threshold=np.float64(0.5), f1=np.float64(0.811))], overall\n",
}


def test_scoring_runs_in_separate_interpreter(tmp_path, monkeypatch):
    stub = tmp_path / "stub"
    for rel, text in STUB.items():
        (stub / rel).parent.mkdir(parents=True, exist_ok=True)
        (stub / rel).write_text(text)
    preds, gts = tmp_path / "pred", tmp_path / "gt"
    preds.mkdir()
    gts.mkdir()
    items = []
    for i in range(3):
        img = np.zeros((20, 30), np.uint8)
        img[10, 5:25] = 255
        cv2.imwrite(str(preds / f"im{i}.png"), img)
        cv2.imwrite(str(gts / f"im{i}.png"), img)
        items.append({"name": f"im{i}", "gt": str(gts / f"im{i}.png")})
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(stub), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setattr(evaluation, "_SCORER", sys.executable)   # as if the installer chose the fallback
    scores, per_sample, curve = evaluation.evaluate(preds, items, max_dist=0.0075, thresholds=99, nproc=1)
    assert scores["ODS"] == 0.811 and scores["OIS"] == 0.83 and scores["AP"] == 0.85
    assert scores["n_images"] == 3 and [s["name"] for s in per_sample] == ["im0", "im1", "im2"]
    assert per_sample[0]["f1"] == 2.0          # pred.max() (1.0) + one ground-truth map, computed in the child
    assert curve[0]["f1"] == 0.811


def test_marker_selects_the_separate_interpreter(tmp_path, monkeypatch):
    marker = tmp_path / "pedger_scorer.json"
    marker.write_text('{"python": "%s"}' % sys.executable.replace("\\", "\\\\"))
    monkeypatch.setattr(evaluation, "SCORER_MARKER", marker)
    monkeypatch.setattr(evaluation, "_SCORER", evaluation._UNSET)
    monkeypatch.setattr(evaluation, "_have_pyedgeeval", lambda: False)
    assert evaluation.ensure_pyedgeeval() == sys.executable


def test_scorer_side_files_are_python38_compatible():
    """These files run under the Python 3.8 scorer: no newer syntax, no torch imports."""
    for rel in ("pedger_kd/evaluation.py", "pedger_kd/score_worker.py", "eval/agreement_bins.py",
                "pedger_kd/__init__.py"):
        src = (ROOT / rel).read_text()
        ast.parse(src, feature_version=(3, 8))
        assert not re.search(r"^\s*(import torch|from torch|from pedger_kd\.(data|train_kd|inference|augment)\b)",
                             src, re.M), rel
        assert not re.search(r"\.removeprefix\(|\.removesuffix\(|\bmatch\s+\w+\s*:", src), rel
