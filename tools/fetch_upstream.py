"""Clone the official upstream repositories into third_party/ at pinned commits.

PEdger is required for everything (student model, LR scheduler, released checkpoint).
The teacher repos are only needed in the teacher-caching notebook.

    python tools/fetch_upstream.py                 # PEdger only
    python tools/fetch_upstream.py --teachers      # + SAUGE, MuGE (UAED repo), MEMO
    python tools/fetch_upstream.py --all           # + DiffusionEdge (backup teacher)
"""
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"

# name: (url, commit) -- commits pinned on 2026-11-12
REPOS = {
    "PEdger": ("https://github.com/ForawardStar/PEdger.git",
               "9c3eeb06b36dcc4322b49eab8f901b19d5809f1a"),
    "SAUGE": ("https://github.com/Star-xing1/SAUGE.git",
              "63a258126e3ac27adf3a12d7b35dd77b8ffe1d1d"),
    "UAED": ("https://github.com/ZhouCX117/UAED.git",  # contains MuGE
             "fb1147d692c9e16be8997337a0c09731ca8a6c2a"),
    "MEMO_Edge_Detection": ("https://github.com/cplusx/MEMO_Edge_Detection.git",
                            "b91b5dcbb69dc86b6174bdb10d58bc1faff1cfc7"),
    "DiffusionEdge": ("https://github.com/GuHuangAI/DiffusionEdge.git",
                      "b03617e84b79336fd1f4d650105580e026ab26a7"),
}
TEACHERS = ["SAUGE", "UAED", "MEMO_Edge_Detection"]


def run(cmd, cwd=None):
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def fetch(name):
    url, commit = REPOS[name]
    dest = THIRD_PARTY / name
    if not dest.exists():
        THIRD_PARTY.mkdir(exist_ok=True)
        run(["git", "init", "-q", str(dest)])
        run(["git", "remote", "add", "origin", url], cwd=dest)
    run(["git", "fetch", "-q", "--depth", "1", "origin", commit], cwd=dest)
    run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=dest)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest, capture_output=True,
                          text=True, check=True).stdout.strip()
    assert head == commit, f"{name}: expected {commit}, got {head}"
    print(f"{name} @ {commit[:10]} -> {dest}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teachers", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    names = ["PEdger"]
    if args.teachers or args.all:
        names += TEACHERS
    if args.all:
        names += ["DiffusionEdge"]
    for n in names:
        fetch(n)


if __name__ == "__main__":
    main()
