"""Install the BSDS benchmark port (pyEdgeEval 0.2.8) on any Python, including 3.13+ (Kaggle).

`pip install pyEdgeEval==0.2.8` fails on Python >= 3.13. This script tries, in order:

  1. nothing to do: pyEdgeEval already works in this Python, or a scorer environment from step 3
     already exists (self-test passes);
  2. COMPILE: download the official 0.2.8 source (SHA-256 pinned), apply small build fixes
     (setup.py version lookup broken by PEP 667 in Python 3.13; C++14 mode; standard #includes
     that new compilers no longer pull in implicitly; a SciPy import removed in SciPy 2.0), build it
     with this Python, and self-test. On failure the compiler's error lines are printed;
  3. FALLBACK (no compiler needed): create a small separate Python 3.8 environment with `uv` and
     install the official prebuilt pyEdgeEval 0.2.8 wheel (published for Python 3.8 only).
     The scoring code then runs the benchmark in that interpreter (pedger_kd/score_worker.py).
The matching algorithm itself is never modified. A self-test checks pixel matching on known
shapes and a perfect-prediction benchmark before anything is used.

    python tools/install_pyedgeeval.py [--mode auto|compile|py38] [--force]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import urllib.request
from pathlib import Path

VERSION = "0.2.8"
URL = ("https://files.pythonhosted.org/packages/77/b9/a7294370ebccd4b9b2fc3f17f8e3578a1e11da7188f8bcd23ab139f27e50/"
       "pyedgeeval-0.2.8.tar.gz")
SHA256 = "dd94dbef7acd499c7266704fa34c01a1fcecab38935253db1b3748849890403c"

# Where the fallback scorer environment is recorded (read by pedger_kd/evaluation.py)
MARKER = Path.home() / ".cache" / "pedger_scorer.json"
PY38_DIR = Path("/kaggle/working/scorer_py38") if Path("/kaggle").exists() else Path.home() / ".cache" / "pedger_scorer_py38"
PY38_PACKAGES = [f"pyEdgeEval=={VERSION}", "numpy<1.25", "scipy<1.11", "opencv-python-headless", "pyyaml"]

# (file, old, new, minimum number of occurrences)
PATCHES = [
    ("setup.py", 'PKG_NAME = "pyEdgeEval"\n', 'PKG_NAME = "pyEdgeEval"\n_INFO = {}  # patched: PEP 667 (Python 3.13)\n', 1),
    ("setup.py", 'exec(compile(f.read(), version_file, "exec"))', 'exec(compile(f.read(), version_file, "exec"), _INFO)', 5),
    ("setup.py", 'return locals()["', 'return _INFO["', 5),
    # C++14 mode; and csa.hh calls fprintf(stderr, msg) with a non-literal format, which Ubuntu's
    # Python build flags turn into an error (-Werror=format-security). These args come after those
    # flags on the compiler command line, so they win.
    ("setup.py", 'extra_compile_args=["-fPIC", "-DNOBLAS"]',
     'extra_compile_args=["-fPIC", "-DNOBLAS", "-std=c++14", "-Wno-error=format-security", "-Wno-format-security"]', 2),
    ("pyEdgeEval/utils/distance_transforms.py", "from scipy.ndimage.morphology import distance_transform_edt",
     "from scipy.ndimage import distance_transform_edt", 1),
    # headers that use the C/C++ standard library without including it (newer libstdc++ no longer
    # includes these transitively)
    ("pyEdgeEval/_lib/include/Matrix.hh", "#define __Matrix_hh__\n",
     "#define __Matrix_hh__\n#include <cstdlib>  // patched\n#include <cmath>  // patched\n", 1),
    ("pyEdgeEval/_lib/include/csa_defs.h", "#define\tTRUE\t1\n", "#include <cmath>  // patched\n#define\tTRUE\t1\n", 1),
    # same output, but with a literal format string (-Werror=format-security)
    ("pyEdgeEval/_lib/include/csa.hh", "fprintf(stderr,nomem_msg)", 'fprintf(stderr, "%s", nomem_msg)', 4),
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
# Perfect prediction on a BSDS-sized image (321x481, match radius ~4 px): precision and recall ~1.
# The official matcher is approximate (sparse graph + random outlier links), so it may drop a
# pixel or so; on a tiny image with a sub-pixel radius it drops more. Allow 1%.
gt = np.zeros((321, 481), bool); gt[100, 40:440] = True; gt[150:280, 300] = True
cr, sr, cp, sp = evaluate_boundaries_threshold_multiple_gts(
    thresholds=np.array([0.5]), pred=gt.astype(np.float64), gts=[gt, gt], max_dist=0.0075,
    apply_thinning=True, apply_nms=False)
assert sp[0] > 0 and cp[0] >= 0.99 * sp[0] and cr[0] >= 0.99 * sr[0], ("perfect prediction", cr, sr, cp, sp)
# A prediction far from every boundary must score ~0
bad = np.zeros_like(gt); bad[250, 40:200] = True
cr, sr, cp, sp = evaluate_boundaries_threshold_multiple_gts(
    thresholds=np.array([0.5]), pred=bad.astype(np.float64), gts=[gt, gt], max_dist=0.0075,
    apply_thinning=True, apply_nms=False)
assert cp[0] <= 0.01 * sp[0], ("far prediction matched", cr, sr, cp, sp)
print("pyEdgeEval self-test passed")
'''


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def self_test(python=None):
    r = run([python or sys.executable, "-c", SELF_TEST])
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def recorded_scorer():
    try:
        py = json.loads(MARKER.read_text())["python"]
        return py if Path(py).exists() else None
    except Exception:
        return None


def record_scorer(python):
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_text(json.dumps({"python": str(python), "pyEdgeEval": VERSION}))


def forget_scorer():
    MARKER.unlink(missing_ok=True)


# ------------------------------------------------------------------ step 2: compile
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


def build_diagnostics():
    lines = []
    for exe in ("gcc", "g++", "cc", "c++"):
        p = shutil.which(exe)
        if p:
            v = run([p, "--version"]).stdout.splitlines()
            lines.append(f"{exe}: {v[0] if v else p}")
        else:
            lines.append(f"{exe}: NOT FOUND")
    inc = sysconfig.get_paths().get("include", "")
    lines.append(f"Python.h: {'found' if Path(inc, 'Python.h').exists() else 'NOT FOUND'} ({inc})")
    return lines


def compile_errors(log):
    errs = [l for l in log.splitlines() if re.search(r"(fatal )?error:|Error:|No such file", l)]
    seen, out = set(), []
    for l in errs:
        if l not in seen:
            seen.add(l)
            out.append(l)
    return out[:40] or log.splitlines()[-60:]


def try_compile(workdir):
    work = Path(workdir)
    tgz = work / f"pyedgeeval-{VERSION}.tar.gz"
    print(f"[compile] downloading pyEdgeEval {VERSION} source ...", flush=True)
    download(tgz)
    with tarfile.open(tgz) as tf:
        tf.extractall(work, filter="data") if sys.version_info >= (3, 12) else tf.extractall(work)
    src = work / f"pyedgeeval-{VERSION}"
    patch_tree(src)
    print("[compile] building (about 1-2 minutes) ...", flush=True)
    r = run([sys.executable, "-m", "pip", "install", "-v", "--no-cache-dir", str(src)])
    if r.returncode != 0:
        print("[compile] build FAILED. Environment:")
        for l in build_diagnostics():
            print("   ", l)
        print("[compile] compiler/build errors:")
        for l in compile_errors(r.stdout + "\n" + r.stderr):
            print("   ", l[:300])
        return False
    ok, out = self_test()
    print("[compile]", out.splitlines()[-1] if out else "")
    return ok


# ------------------------------------------------------------------ step 3: prebuilt wheel in Python 3.8
def uv_cmd():
    if shutil.which("uv"):
        return [shutil.which("uv")]
    r = run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    if r.returncode != 0:
        raise RuntimeError("could not install uv: " + r.stderr[-500:])
    return [sys.executable, "-m", "uv"]


def setup_py38():
    uv = uv_cmd()
    py = PY38_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not py.exists():
        print(f"[py38] creating a Python 3.8 environment in {PY38_DIR} (uv downloads Python 3.8) ...", flush=True)
        r = run(uv + ["venv", "--python", "3.8", str(PY38_DIR)])
        if r.returncode != 0:
            raise RuntimeError("uv could not create a Python 3.8 environment:\n" + (r.stdout + r.stderr)[-1500:])
    print("[py38] installing the prebuilt pyEdgeEval wheel ...", flush=True)
    r = run(uv + ["pip", "install", "--python", str(py)] + PY38_PACKAGES)
    if r.returncode != 0:
        raise RuntimeError("installing pyEdgeEval into Python 3.8 failed:\n" + (r.stdout + r.stderr)[-1500:])
    return py


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["auto", "compile", "py38"], default="auto")
    p.add_argument("--force", action="store_true", help="redo the installation even if a working scorer exists")
    p.add_argument("--workdir", default=None)
    args = p.parse_args()

    if not args.force:
        ok, _ = self_test()
        if ok:
            print("pyEdgeEval already installed and working (this Python)")
            return
        py = recorded_scorer()
        if py and self_test(py)[0]:
            print(f"pyEdgeEval scorer environment already set up: {py}")
            return

    if args.mode in ("auto", "compile"):
        work = Path(args.workdir or tempfile.mkdtemp(prefix="pyedgeeval_"))
        work.mkdir(parents=True, exist_ok=True)
        try:
            if try_compile(work):
                forget_scorer()
                print("pyEdgeEval self-test passed (compiled for this Python)")
                return
        except Exception as e:
            print(f"[compile] failed: {type(e).__name__}: {e}")
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if args.mode == "compile":
            sys.exit("PROBLEM: compiling pyEdgeEval failed (errors above).")
        print("[compile] did not work; switching to the prebuilt-wheel fallback (no compiler needed)", flush=True)

    try:
        py = setup_py38()
    except Exception as e:
        sys.exit(f"PROBLEM: pyEdgeEval fallback setup failed: {e}")
    ok, out = self_test(str(py))
    print("[py38]", out.splitlines()[-1] if out else "")
    if not ok:
        print(out[-3000:])
        sys.exit("PROBLEM: the Python 3.8 scorer environment failed its self-test (output above).")
    record_scorer(py)
    print(f"pyEdgeEval self-test passed (prebuilt wheel in {py}); scoring will run there automatically")


if __name__ == "__main__":
    main()
