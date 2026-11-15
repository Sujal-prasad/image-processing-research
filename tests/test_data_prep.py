"""Data preparation on a fake BSDS500 release in the original Berkeley format."""
import json
import subprocess
import sys

import cv2
import numpy as np
import scipy.io as sio
import torch
from conftest import ROOT

from pedger_kd.data import ArrayStore
from pedger_kd.evaluation import load_gt


def write_bsds_mat(path, boundaries):
    gt = np.empty((1, len(boundaries)), dtype=object)
    for i, b in enumerate(boundaries):
        gt[0, i] = {"Boundaries": b.astype(np.uint8), "Segmentation": np.ones_like(b, np.uint16)}
    sio.savemat(path, {"groundTruth": gt})


def fake_bsds(root):
    rng = np.random.default_rng(0)
    for split, n in (("train", 3), ("val", 2), ("test", 2)):
        (root / "images" / split).mkdir(parents=True)
        (root / "groundTruth" / split).mkdir(parents=True)
        for i in range(n):
            H, W = (48, 72) if i % 2 == 0 else (72, 48)          # mix landscape / portrait like BSDS
            cv2.imwrite(str(root / "images" / split / f"{split}{i}.jpg"),
                        rng.integers(0, 255, (H, W, 3), dtype=np.uint8))
            bs = []
            for k in range(5):                                  # 5 annotators, partly agreeing
                b = np.zeros((H, W), bool)
                b[H // 2, :] = True
                if k < 2:
                    b[:, W // 3] = True
                bs.append(b)
            write_bsds_mat(root / "groundTruth" / split / f"{split}{i}.mat", bs)


def test_prepare_bsds_end_to_end(tmp_path):
    raw = tmp_path / "BSR" / "BSDS500" / "data"
    fake_bsds(raw)
    gts = load_gt(raw / "groundTruth" / "train" / "train0.mat")
    assert len(gts) == 5 and gts[0].dtype == bool

    store = tmp_path / "store" / "bsds"
    subprocess.run([sys.executable, str(ROOT / "data" / "prepare_bsds.py"), "--bsds-root", str(raw),
                    "--out", str(store)], check=True, cwd=ROOT)
    meta = json.load(open(store / "meta.json"))
    assert meta["splits"]["train"]["annotators_max"] == 5
    assert len(json.load(open(store / "eval_lists" / "test.json"))) == 2

    st = ArrayStore(store, "trainval", device="cpu")
    assert len(st) == 5 and tuple(st.images.shape[1:]) == (48, 72, 3)   # all landscape
    frac = st.frac[0].float()
    assert torch.isclose(frac[24, 5], torch.tensor(1.0))                # all 5 annotators
    transposed = json.load(open(store / "trainval" / "transposed.json"))
    assert transposed == [False, True, False, False, True]

    # teachers: export in original orientation, "predict" = the annotator fraction, pack back
    imgs = tmp_path / "img"
    subprocess.run([sys.executable, str(ROOT / "teachers" / "export_images.py"), "--store", str(store),
                    "--split", "trainval", "--out", str(imgs)], check=True, cwd=ROOT)
    names = json.load(open(store / "trainval" / "names.json"))
    pred = tmp_path / "pred"
    pred.mkdir()
    for n in names:
        split = "train" if n.startswith("train") else "val"
        a = np.mean(np.stack(load_gt(raw / "groundTruth" / split / f"{n}.mat")), 0)
        exported = cv2.imread(str(imgs / f"{n}.png"))
        assert exported.shape[:2] == a.shape                        # exported in original orientation
        cv2.imwrite(str(pred / f"{n}.png"), (a * 65535).astype(np.uint16))
    cache = tmp_path / "cache"
    subprocess.run([sys.executable, str(ROOT / "teachers" / "pack_cache.py"), "--store", str(store),
                    "--split", "trainval", "--teacher", "fake", "--pred-dir", str(pred), "--out", str(cache)],
                   check=True, cwd=ROOT)
    st2 = ArrayStore(store, "trainval", cache, "fake", device="cpu")
    # teacher map must line up with the annotator fraction after re-orientation
    assert torch.allclose(st2.teacher.float() / 255, st2.frac.float(), atol=1 / 255 + 1e-3)
    # a teacher step that did not finish: packing names the missing predictions
    (pred / f"{names[-1]}.png").unlink()
    r = subprocess.run([sys.executable, str(ROOT / "teachers" / "pack_cache.py"), "--store", str(store),
                        "--split", "trainval", "--teacher", "fake", "--pred-dir", str(pred), "--out", str(cache)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and f"1 of {len(names)} predictions are missing" in r.stderr

    # step 1 path: predictions from the released PEdger checkpoint (scoring needs pyEdgeEval, on Kaggle)
    ckpt = ROOT / "third_party" / "PEdger" / "checkpoint_PEdger.pth"
    if ckpt.exists():
        for mode in ("ns_m", "fuse_m"):
            pdir = tmp_path / f"released_{mode}"
            subprocess.run([sys.executable, str(ROOT / "eval" / "run_bsds_eval.py"), "--ckpt", str(ckpt),
                            "--mode", mode, "--ms", "--store", str(store), "--split", "test", "--raw-root", str(raw),
                            "--out-dir", str(pdir), "--predict-only", "--device", "cpu"], check=True, cwd=ROOT)
            for it in json.load(open(store / "eval_lists" / "test.json")):
                p = cv2.imread(str(pdir / f"{it['name']}.png"), cv2.IMREAD_UNCHANGED)
                img = cv2.imread(str(raw / it["image"]))
                assert p.dtype == np.uint8 and p.shape == img.shape[:2]

    out = tmp_path / "vis.png"
    subprocess.run([sys.executable, str(ROOT / "data" / "visualize_samples.py"), "--store", str(store),
                    "--split", "trainval", "--n", "3", "--teacher-root", str(cache), "--teacher", "fake",
                    "--augmented", "--out", str(out)], check=True, cwd=ROOT)
    assert cv2.imread(str(out)) is not None
