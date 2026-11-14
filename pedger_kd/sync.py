"""Remote storage that outlives Kaggle sessions and Kaggle accounts.

All persistent state (array stores, teacher caches, run checkpoints, scores, code)
lives in ONE private Hugging Face dataset repo. A Kaggle notebook on any account
that has the same `HF_TOKEN` secret can pull it, continue, and push back.

The remote mirrors a local "workspace" folder (paths are identical):
    data/<store>/...          array stores        (data/prepare_*.py)
    teachers/<dir>/...        teacher caches      (teachers/pack_cache.py)
    runs/<run>/...            last.pt, log.jsonl, config.yaml, progress.json, final.pt, done.json, scores_*.json
    code/...                  optional copy of this repo (if you don't use GitHub)
    timing/, results/         timing estimates, tables

Remote spec strings: "hf:<user>/<repo>" (token from $HF_TOKEN) or "local:<folder>"
(a folder acting as the remote; used by the tests, also works with a mounted drive).

CLI (from a laptop or a notebook):
    python -m pedger_kd.sync --remote hf:you/pedger-store status
    python -m pedger_kd.sync --remote hf:you/pedger-store push-dir --local data_store/bsds --to data/bsds
    python -m pedger_kd.sync --remote hf:you/pedger-store pull --prefix runs/A_agree_s0 --ws /kaggle/working/ws
    python -m pedger_kd.sync --remote hf:you/pedger-store push-code --stamp   # where the code is edited
    python -m pedger_kd.sync --remote hf:you/pedger-store squash      # drop old checkpoint versions
"""
import argparse
import datetime as dt
import fnmatch
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")   # keep Kaggle logs readable


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


def iso(t=None):
    return (t or utcnow()).isoformat(timespec="seconds")


def age_minutes(iso_str):
    try:
        return (utcnow() - dt.datetime.fromisoformat(iso_str)).total_seconds() / 60
    except Exception:
        return float("inf")


class LocalRemote:
    """A folder acting as the remote (tests, mounted drives)."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.spec = f"local:{self.root}"

    def list(self, prefix=""):
        base = self.root / prefix
        if base.is_file():
            return [prefix]
        if not base.exists():
            return []
        return sorted(p.relative_to(self.root).as_posix() for p in base.rglob("*") if p.is_file())

    def exists(self, path):
        return (self.root / path).is_file()

    def read_json(self, path):
        p = self.root / path
        return json.loads(p.read_text()) if p.is_file() else None

    def download(self, prefix, ws):
        n = 0
        for rel in self.list(prefix):
            dst = Path(ws) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.root / rel, dst)
            n += 1
        return n

    def upload(self, files, message=""):
        """files: {remote_path: local_path}"""
        for rp, lp in files.items():
            dst = self.root / rp
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".tmp")
            shutil.copy2(lp, tmp)
            os.replace(tmp, dst)

    def upload_dir(self, local_dir, remote_prefix, ignore=()):
        files = {}
        for p in Path(local_dir).rglob("*"):
            rel = p.relative_to(local_dir).as_posix()
            if p.is_file() and not any(fnmatch.fnmatch(rel, pat) for pat in ignore):
                files[f"{remote_prefix}/{rel}"] = str(p)
        self.upload(files)
        return len(files)

    def squash(self):
        pass


class HubRemote:
    """A private Hugging Face dataset repo."""

    def __init__(self, repo_id, token=None):
        from huggingface_hub import HfApi
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN")
        if not self.token:
            raise RuntimeError("HF_TOKEN is not set (add it as a Kaggle secret, see README)")
        self.api = HfApi(token=self.token)
        self.api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        self.spec = f"hf:{repo_id}"

    def _retry(self, fn, what, attempts=4):
        for a in range(attempts):
            try:
                return fn()
            except Exception as e:
                if a == attempts - 1:
                    raise
                wait = 10 * (a + 1)
                print(f"remote {what} failed ({e}); retrying in {wait}s", flush=True)
                time.sleep(wait)

    def list(self, prefix=""):
        files = self._retry(lambda: self.api.list_repo_files(self.repo_id, repo_type="dataset"), "list")
        return sorted(f for f in files if f == prefix or f.startswith(prefix.rstrip("/") + "/") or not prefix)

    def exists(self, path):
        return self._retry(lambda: self.api.file_exists(self.repo_id, path, repo_type="dataset"), "exists")

    def read_json(self, path):
        from huggingface_hub import hf_hub_download
        if not self.exists(path):
            return None
        with tempfile.TemporaryDirectory() as tmp:
            p = self._retry(lambda: hf_hub_download(self.repo_id, path, repo_type="dataset", token=self.token,
                                                    local_dir=tmp), "read")
            return json.loads(Path(p).read_text())

    def download(self, prefix, ws):
        from huggingface_hub import snapshot_download
        files = self.list(prefix)
        if not files:
            return 0
        self._retry(lambda: snapshot_download(self.repo_id, repo_type="dataset", token=self.token,
                                              allow_patterns=[f"{prefix.rstrip('/')}/*", prefix],
                                              local_dir=str(ws)), "download")
        return len(files)

    def upload(self, files, message="update"):
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(path_in_repo=rp, path_or_fileobj=str(lp)) for rp, lp in files.items()]
        self._retry(lambda: self.api.create_commit(self.repo_id, operations=ops, commit_message=message,
                                                   repo_type="dataset"), "upload")

    def upload_dir(self, local_dir, remote_prefix, ignore=()):
        self._retry(lambda: self.api.upload_folder(folder_path=str(local_dir), path_in_repo=remote_prefix,
                                                   repo_id=self.repo_id, repo_type="dataset",
                                                   ignore_patterns=list(ignore) or None,
                                                   commit_message=f"upload {remote_prefix}"), "upload_dir")
        return sum(1 for p in Path(local_dir).rglob("*") if p.is_file() and not any(
            fnmatch.fnmatch(p.relative_to(local_dir).as_posix(), pat) for pat in ignore))

    def squash(self):
        """Drop old file versions (every periodic checkpoint creates one) to keep storage small."""
        self._retry(lambda: self.api.super_squash_history(self.repo_id, repo_type="dataset"), "squash")


def open_remote(spec):
    if not spec:
        return None
    kind, _, target = spec.partition(":")
    if kind == "hf":
        return HubRemote(target)
    if kind == "local":
        return LocalRemote(target)
    raise ValueError(f"remote spec must be hf:<repo> or local:<dir>, got {spec!r}")


class AsyncPusher:
    """Uploads in a background thread so training never waits for the network.

    Requests are coalesced: if several checkpoints are requested while an upload is
    running, only the newest is uploaded next. Files are copied to a staging folder
    first, so the trainer can keep overwriting last.pt safely.
    """

    def __init__(self, remote, staging_dir):
        self.remote = remote
        self.staging = Path(staging_dir)
        self.staging.mkdir(parents=True, exist_ok=True)
        self._pending = None
        self._cv = threading.Condition()
        self._busy = False
        self._counter = 0
        self.errors = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def request(self, files, message):
        with self._cv:
            self._counter += 1
            d = self.staging / f"{self._counter:06d}"
            d.mkdir(parents=True, exist_ok=True)
            staged = {}
            for rp, lp in files.items():
                if Path(lp).exists():
                    dst = d / rp.replace("/", "__")
                    shutil.copy2(lp, dst)
                    staged[rp] = str(dst)
            if self._pending is not None:
                shutil.rmtree(self._pending[2], ignore_errors=True)
            self._pending = (staged, message, d)
            self._cv.notify_all()

    def _loop(self):
        while True:
            with self._cv:
                while self._pending is None:
                    self._cv.wait()
                staged, message, d = self._pending
                self._pending = None
                self._busy = True
            try:
                if staged:
                    self.remote.upload(staged, message)
            except Exception as e:
                self.errors += 1
                print(f"WARNING: upload failed, will retry with the next checkpoint: {e}", flush=True)
            finally:
                shutil.rmtree(d, ignore_errors=True)
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()

    def flush(self, timeout=1800):
        end = time.time() + timeout
        with self._cv:
            while (self._pending is not None or self._busy) and time.time() < end:
                self._cv.wait(timeout=5)
            return self._pending is None and not self._busy


STAMP_FILE = "CODE_STAMP.json"


def write_stamp(repo_root=ROOT):
    """Mark this copy of the code as the newest version. Done only where the code is edited
    (`push-code --stamp`, tools/make_code_zip.py), never in a notebook."""
    import subprocess
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, capture_output=True,
                             text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        git = "unknown"
    stamp = {"stamp": iso(), "git": git}
    (Path(repo_root) / STAMP_FILE).write_text(json.dumps(stamp, indent=2))
    return stamp


def read_stamp(repo_root=ROOT):
    p = Path(repo_root) / STAMP_FILE
    return json.loads(p.read_text()) if p.exists() else None


def push_code(remote, repo_root=ROOT, stamp=False, force=False):
    """Upload the code to <remote>/code, but never replace newer code with an older copy.

    Each pushed copy carries CODE_STAMP.json. A copy is uploaded only if its stamp is newer than the
    remote's, so a notebook session holding an old download cannot overwrite a fix pushed meanwhile.
    """
    if stamp:
        write_stamp(repo_root)
    local = read_stamp(repo_root)
    remote_stamp = remote.read_json(f"code/{STAMP_FILE}")
    if not force:
        if local is None:
            print(f"not pushing code: this copy has no {STAMP_FILE} (run `push-code --stamp` where the code is edited)")
            return False
        if remote_stamp and remote_stamp.get("stamp", "") >= local["stamp"]:
            print(f"not pushing code: the remote copy ({remote_stamp.get('stamp')}) is the same as or newer than "
                  f"this one ({local['stamp']})")
            return False
    ignore = ["third_party/*", "dist/*", "runs/*", "data_store/*", "teacher_cache/*", "outputs/*", "ws/*",
              "*__pycache__*", ".git/*", "*.pt", "*.pth", "*.npy", ".pytest_cache/*", "*.tmp"]
    n = remote.upload_dir(repo_root, "code", ignore=ignore)
    print(f"pushed {n} code files to {remote.spec}/code (stamp {(local or {}).get('stamp')})")
    return True


def status_table(remote):
    runs = sorted({f.split("/")[1] for f in remote.list("runs") if f.count("/") >= 2})
    print(f"{'run':38s} {'status':9s} {'epoch':>5s} {'step':>7s} {'GPU-h':>6s} {'heartbeat':>12s}  worker")
    for r in runs:
        p = remote.read_json(f"runs/{r}/progress.json") or {}
        hb = p.get("heartbeat")
        hb_s = f"{age_minutes(hb):.0f} min ago" if hb else "-"
        print(f"{r:38s} {p.get('status', '?'):9s} {str(p.get('epoch', '-')):>5s} {str(p.get('step', '-')):>7s} "
              f"{p.get('gpu_hours', 0):6.2f} {hb_s:>12s}  {p.get('worker', '')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--remote", default=os.environ.get("PEDGER_REMOTE"), help="hf:<user>/<repo> or local:<dir>")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("push-dir"); s.add_argument("--local", required=True); s.add_argument("--to", required=True)
    s = sub.add_parser("pull"); s.add_argument("--prefix", required=True); s.add_argument("--ws", required=True)
    s = sub.add_parser("push-code")
    s.add_argument("--stamp", action="store_true", help="mark this copy as newest first (only where the code is edited)")
    s.add_argument("--force", action="store_true", help="upload even if the remote copy is newer")
    sub.add_parser("status")
    s = sub.add_parser("ls"); s.add_argument("--prefix", default="")
    sub.add_parser("squash")
    args = p.parse_args()
    remote = open_remote(args.remote)
    if remote is None:
        p.error("--remote (or $PEDGER_REMOTE) is required")
    if args.cmd == "push-dir":
        n = remote.upload_dir(args.local, args.to.rstrip("/"))
        print(f"pushed {n} files to {args.to}")
    elif args.cmd == "pull":
        print(f"pulled {remote.download(args.prefix, args.ws)} files from {args.prefix}")
    elif args.cmd == "push-code":
        push_code(remote, stamp=args.stamp, force=args.force)
    elif args.cmd == "status":
        status_table(remote)
    elif args.cmd == "ls":
        print("\n".join(remote.list(args.prefix)))
    elif args.cmd == "squash":
        remote.squash()
        print("history squashed")


if __name__ == "__main__":
    main()
