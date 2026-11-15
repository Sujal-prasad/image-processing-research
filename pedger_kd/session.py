"""Kaggle session clock: when did this session start, and by when must long steps stop?

Kaggle stops every notebook session after 12 hours. Long steps (scoring, training) stop a little
earlier, save their progress, and continue in the next run of the notebook.

The start time comes from PEDGER_SESSION_START (set by the notebook setup cell), else from the start
time of the notebook's Jupyter kernel process, else from the container's first process. Outside
Kaggle there is no limit (None). Standard library only (also runs under the Python 3.8 scorer).
"""
import os
import time
from pathlib import Path

KAGGLE_LIMIT_HOURS = 12.0


def _stat_fields(pid):
    """Fields of /proc/<pid>/stat after "pid (comm)", i.e. index 0 = field 3 (state)."""
    stat = Path(f"/proc/{pid}/stat").read_text()
    return stat[stat.rindex(")") + 2:].split()


def _proc_start(pid):
    try:
        ticks = int(_stat_fields(pid)[19])                    # field 22: start time, clock ticks after boot
        btime = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                     if line.startswith("btime"))
        return btime + ticks / os.sysconf("SC_CLK_TCK")
    except Exception:
        return None


def _parent(pid):
    try:
        return int(_stat_fields(pid)[1])                       # field 4: parent pid
    except Exception:
        return None


def _cmdline(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except Exception:
        return ""


def on_kaggle():
    return Path("/kaggle").exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def session_start():
    env = os.environ.get("PEDGER_SESSION_START")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    if not on_kaggle():
        return None
    pid = os.getpid()
    for _ in range(30):                                        # walk up to the notebook kernel
        pid = _parent(pid)
        if not pid or pid <= 1:
            break
        if "ipykernel" in _cmdline(pid):
            return _proc_start(pid)
    return _proc_start(1)


def deadline(reserve_minutes=45.0, limit_hours=KAGGLE_LIMIT_HOURS):
    """Epoch time by which long steps should stop, leaving `reserve_minutes` for uploads and the rest of
    the notebook. None when there is no session limit (not on Kaggle, or the start time is unknown)."""
    start = session_start()
    return None if start is None else start + limit_hours * 3600 - reserve_minutes * 60


def minutes_left(dl):
    return float("inf") if dl is None else (dl - time.time()) / 60
