"""Run pedger_kd.evaluation.evaluate() inside the separate scorer interpreter (see tools/install_pyedgeeval.py).

    python -m pedger_kd.score_worker spec.json out.json
Must stay compatible with Python 3.8 and must not import torch.
"""
import json
import sys

from pedger_kd.evaluation import evaluate


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
    scores, per_sample, curve = evaluate(**spec)
    with open(out_path, "w") as f:
        json.dump(plain({"scores": scores, "per_sample": per_sample, "curve": curve}), f)


if __name__ == "__main__":
    main()
