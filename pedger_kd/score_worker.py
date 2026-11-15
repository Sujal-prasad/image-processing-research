"""Scoring worker: runs pedger_kd.evaluation.score_here() in a fresh process (started by evaluate()).

    python -m pedger_kd.score_worker spec.json out.json
It runs under this Python, or under the separate Python 3.8 scorer environment
(see tools/install_pyedgeeval.py). Must stay compatible with Python 3.8 and must not import torch.
"""
import json
import sys

from pedger_kd.evaluation import score_here


def plain(x):
    """numpy scalars/arrays -> JSON-serialisable Python values."""
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def main():
    spec_path, out_path = sys.argv[1], sys.argv[2]
    with open(spec_path) as f:
        spec = json.load(f)
    res = score_here(**spec)
    with open(out_path, "w") as f:
        json.dump(plain(res), f)


if __name__ == "__main__":
    main()
