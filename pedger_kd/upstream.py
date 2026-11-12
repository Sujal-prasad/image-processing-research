"""Load the official, unmodified PEdger code from third_party/PEdger.

PEdger's files use generic module names (models, utils, ...), so they are loaded
by file path under private names to avoid clashing with anything else.

Facts verified against the official code (commit 9c3eeb0):
  * Guider_stu      = recurrent network,      364,685 params (~0.36M)
  * Guider_noshare  = non-recurrent network,  734,225 params (~0.73M, the
                      size PEdger reports; test_speed.py times this one)
  * every BatchNorm2d uses track_running_stats=False, i.e. it ALWAYS normalises
    with the statistics of the current batch, at train and at test time
  * training uses batch_size=1 and iter_size=16 (gradient accumulation)
"""
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PEDGER_DIR = ROOT / "third_party" / "PEdger"

_cache = {}


def pedger_dir():
    d = Path(os.environ.get("PEDGER_DIR", DEFAULT_PEDGER_DIR))
    if not (d / "models.py").exists():
        raise FileNotFoundError(
            f"Official PEdger code not found at {d}. Run: python tools/fetch_upstream.py")
    return d


def _load(filename, modname):
    if modname in _cache:
        return _cache[modname]
    path = pedger_dir() / filename
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    _cache[modname] = mod
    return mod


def models():
    return _load("models.py", "_pedger_models")


def models_noshare():
    return _load("models_noshare.py", "_pedger_models_noshare")


def utils():
    return _load("utils.py", "_pedger_utils")


def loss_function():
    return _load("loss_function.py", "_pedger_loss_function")


def build_recurrent():
    return models().Guider_stu()


def build_noshare():
    return models_noshare().Guider_noshare()


def released_checkpoint_path():
    return pedger_dir() / "checkpoint_PEdger.pth"
