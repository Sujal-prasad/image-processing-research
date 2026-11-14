"""The pyEdgeEval patcher (offline): it fixes the Python 3.13 build bug and fails loudly on unexpected sources."""
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import ROOT

sys.path.insert(0, str(ROOT / "tools"))
import install_pyedgeeval as ip  # noqa: E402

# The parts of pyEdgeEval 0.2.8 the patches touch, verbatim
SETUP_PY = '''PKG_NAME = "pyEdgeEval"
import os.path as osp
def _find(key):
    version_file = osp.join(PKG_NAME, "info.py")
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["%s" % key]
def find_version():
    version_file = osp.join(PKG_NAME, "info.py")
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"]
def find_author():
    version_file = osp.join(PKG_NAME, "info.py")
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__author__"]
def find_email():
    version_file = osp.join(PKG_NAME, "info.py")
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__email__"]
def find_url():
    version_file = osp.join(PKG_NAME, "info.py")
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__url__"]
ARGS = [dict(extra_compile_args=["-fPIC", "-DNOBLAS"]), dict(extra_compile_args=["-fPIC", "-DNOBLAS"])]
print(find_version(), find_author(), " ".join(ARGS[0]["extra_compile_args"][2:]))
'''


def make_tree(tmp_path):
    (tmp_path / "pyEdgeEval" / "utils").mkdir(parents=True)
    (tmp_path / "setup.py").write_text(SETUP_PY)
    (tmp_path / "pyEdgeEval" / "info.py").write_text(
        '__version__ = "0.2.8"\n__author__ = "x"\n__email__ = "y"\n__url__ = "z"\n')
    (tmp_path / "pyEdgeEval" / "utils" / "distance_transforms.py").write_text(
        "from scipy.ndimage.morphology import distance_transform_edt\n")
    inc = tmp_path / "pyEdgeEval" / "_lib" / "include"
    inc.mkdir(parents=True)
    (inc / "Matrix.hh").write_text("\n#ifndef __Matrix_hh__\n#define __Matrix_hh__\nclass Matrix {};\n#endif\n")
    (inc / "csa_defs.h").write_text("#define\tTRUE\t1\n#define\tFALSE\t0\n")
    (inc / "csa.hh").write_text("".join("if (!s) {\n  (void) fprintf(stderr,nomem_msg);\n}\n" for _ in range(4)))
    return tmp_path


def test_patched_setup_runs_and_uses_cpp14(tmp_path):
    src = make_tree(tmp_path)
    ip.patch_tree(src)
    out = subprocess.run([sys.executable, "setup.py"], cwd=src, capture_output=True, text=True, check=True).stdout
    assert out.split() == ["0.2.8", "x", "-std=c++14", "-Wno-error=format-security", "-Wno-format-security"]
    assert "from scipy.ndimage import distance_transform_edt" in \
        (src / "pyEdgeEval/utils/distance_transforms.py").read_text()
    m = (src / "pyEdgeEval/_lib/include/Matrix.hh").read_text()
    assert m.index("#define __Matrix_hh__") < m.index("#include <cstdlib>") < m.index("class Matrix")
    assert (src / "pyEdgeEval/_lib/include/csa_defs.h").read_text().startswith("#include <cmath>")
    csa = (src / "pyEdgeEval/_lib/include/csa.hh").read_text()
    assert csa.count('fprintf(stderr, "%s", nomem_msg)') == 4 and "fprintf(stderr,nomem_msg)" not in csa


def test_unexpected_source_fails_loudly(tmp_path):
    src = make_tree(tmp_path)
    (src / "setup.py").write_text("print('something else')\n")
    with pytest.raises(RuntimeError, match="patch target not found"):
        ip.patch_tree(src)


def test_pinned_checksum_format():
    assert len(ip.SHA256) == 64 and ip.URL.endswith(f"pyedgeeval-{ip.VERSION}.tar.gz")
