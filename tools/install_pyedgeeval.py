"""Install the BSDS benchmark port (pyEdgeEval 0.2.8) on any Python, including 3.13+ (Kaggle).

`pip install pyEdgeEval==0.2.8` fails on Python >= 3.13: its setup.py reads the version with
`exec(...)` + `locals()[...]` inside a function, which PEP 667 (Python 3.13) changed, so the build
stops with KeyError: '__version__' before compiling anything. This script:
  1. skips everything if a working pyEdgeEval is already installed (self-test passes);
  2. downloads the official 0.2.8 source from PyPI and checks its SHA-256;
  3. applies three small, explicit patches (setup.py version lookup; C++14 mode for the old
     `register` keywords in csa.hh; scipy.ndimage import that SciPy 2.0 removes);
  4. builds and installs it with pip;
  5. runs a self-test: pixel matching on known shapes and a perfect-prediction benchmark.
The algorithm code is not modified.

    python tools/install_pyedgeeval.py [--workdir DIR] [--force]
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

VERSION = "0.2.8"
URL = ("https://files.pythonhosted.org/packages/77/b9/a7294370ebccd4b9b2fc3f17f8e3578a1e11da7188f8bcd23ab139f27e50/"
       "pyedgeeval-0.2.8.tar.gz")
SHA256 = "dd94dbef7acd499c7266704fa34c01a1fcecab38935253db1b3748849890403c"

# (file, old, new, minimum number of occurrences)
PATCHES = [
    ("setup.py", 'PKG_NAME = "pyEdgeEval"\n', 'PKG_NAME = "pyEdgeEval"\n_INFO = {}  # patched: PEP 667 (Python 3.13)\n', 1),
    ("setup.py", 'exec(compile(f.read(), version_file, "exec"))', 'exec(compile(f.read(), version_file, "exec"), _INFO)', 5),
    ("setup.py", 'return locals()["', 'return _INFO["', 5),
    ("setup.py", 'extra_compile_args=["-fPIC", "-DNOBLAS"]', 'extra_compile_args=["-fPIC", "-DNOBLAS", "-std=c++14"]', 2),
    ("pyEdgeEval/utils/distance_transforms.py", "from scipy.ndimage.morphology import distance_transform_edt",
     "from scipy.ndimage import distance_transform_edt", 1),
]

SELF_TEST = r'''
import numpy as np
from pyEdgeEval._lib import correspond_pixels
from pyEdgeEval.common.binary_label import calculate_metrics  # noqa: F401  (import chain check)
from pyEdgeEval.common.binary_label.evaluate_boundaries import evaluate_boundaries_threshold_multiple_gts
from pyEdgeEval.preprocess import binary_thin, fast_nms  # noqa: F401
a = np.zeros((60, 80), bool); a[30, 10:70] = True
near = np.zeros_like(a); near[31, 10:70] = True      # 1 px away: must match
far = np.zeros_like(a); far[5, 10:70] = True         # 25 px away: must not match
m1, m2, _, _ = correspond_pixels(a, near, max_dist=0.05)
assert (m1 > 0).sum() == a.sum() and (m2 > 0).sum() == near.sum(), "near line not matched"
m1, m2, _, _ = correspond_pixels(a, far, max_dist=0.05)
assert (m1 > 0).sum() == 0 and (m2 > 0).sum() == 0, "far line wrongly matched"
pred = a.astype(np.float64)                            # perfect prediction -> precision = recall = 1
cr, sr, cp, sp = evaluate_boundaries_threshold_multiple_gts(
    thresholds=np.array([0.5]), pred=pred, gts=[a, a], max_dist=0.0075,
    apply_thinning=True, apply_nms=False)
assert cr[0] == sr[0] and cp[0] == sp[0] and sp[0] > 0, (cr, sr, cp, sp)
print("pyEdgeEval self-test passed")
'''


def self_test():
    r = subprocess.run([sys.executable, "-c", SELF_TEST], capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def patch_tree(root):
    root = Path(root)
    for rel, old, new, min_count in PATCHES:
        path = root / rel
        text = path.read_text(encoding="utf-8")
        n = text.count(old)
        if n < min_count:
            raise RuntimeError(f"patch target not found in {rel} ({n} < {min_count}): {old!r}. "
                               f"The downloaded source differs from pyEdgeEval {VERSION}.")
        path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"patched {len(PATCHES)} spots in pyEdgeEval {VERSION}")


def download(dest):
    data = urllib.request.urlopen(URL, timeout=120).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"checksum mismatch for {URL}: {digest}")
    Path(dest).write_bytes(data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", default=None)
    p.add_argument("--force", action="store_true", help="reinstall even if a working copy is present")
    args = p.parse_args()
    if not args.force:
        ok, out = self_test()
        if ok:
            print("pyEdgeEval already installed and working")
            return
    work = Path(args.workdir or tempfile.mkdtemp(prefix="pyedgeeval_"))
    work.mkdir(parents=True, exist_ok=True)
    tgz = work / f"pyedgeeval-{VERSION}.tar.gz"
    print(f"downloading pyEdgeEval {VERSION} source ...", flush=True)
    download(tgz)
    with tarfile.open(tgz) as tf:
        tf.extractall(work, filter="data") if sys.version_info >= (3, 12) else tf.extractall(work)
    src = work / f"pyedgeeval-{VERSION}"
    patch_tree(src)
    print("building and installing (compiles C++, about 1-2 minutes) ...", flush=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", str(src)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-4000:])
        sys.exit("PROBLEM: pyEdgeEval build failed (log above).")
    ok, out = self_test()
    print(out)
    if not ok:
        sys.exit("PROBLEM: pyEdgeEval installed but the self-test failed (output above).")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
