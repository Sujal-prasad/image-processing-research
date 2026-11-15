"""The Kaggle notebooks: every code cell is valid Python, and sh() stops the run when a step fails."""
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_notebooks", ROOT / "tools" / "make_notebooks.py")
mn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mn)


def test_every_code_cell_compiles():
    for name, (_, cells) in mn.NOTEBOOKS.items():
        for i, c in enumerate(cells):
            if c["cell_type"] == "code":
                src = re.sub(r"^(\s*)[!%].*$", r"\1pass", c["source"], flags=re.M)   # IPython shell/magic lines
                compile(src, f"{name} cell {i}", "exec")


def test_sh_shows_output_and_stops_on_failure(tmp_path, capsys):
    ns = {"subprocess": subprocess}
    exec(mn.SH, ns)
    step = tmp_path / "step.py"
    step.write_text("import sys\nprint('loading 10%\\rloading 100%')\nprint('done')\nsys.exit(int(sys.argv[1]))\n")
    ns["sh"](f'"{sys.executable}" "{step}" 0')
    out = capsys.readouterr().out
    assert "loading 100%" in out and "loading 10%" not in out and "done" in out
    with pytest.raises(RuntimeError, match="exit code 3"):
        ns["sh"](f'"{sys.executable}" "{step}" 3')
