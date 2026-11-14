"""Preflight: detect the session and check everything a Kaggle run needs, before any work starts.

Detects automatically: Kaggle run type (interactive vs. background commit), GPUs, the Hugging Face
account behind HF_TOKEN, whether it can reach and write the remote repo, and internet access.
Prints one table and exits non-zero with a plain-language fix if something blocks the run.

    python -m pedger_kd.preflight [--remote hf:<user>/<repo>] [--need-gpu]
"""
import argparse
import os
import sys


def kaggle_run_type():
    t = os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
    if t is None:
        return "not Kaggle" if not os.path.exists("/kaggle") else "Kaggle (type unknown)"
    return {"Interactive": "Kaggle interactive (stops when the browser tab closes!)",
            "Batch": "Kaggle background commit (laptop can be off)"}.get(t, f"Kaggle {t}")


def gpus():
    try:
        import torch
        n = torch.cuda.device_count()
        return n, ", ".join(torch.cuda.get_device_name(i) for i in range(n)) or "none"
    except Exception as e:
        return 0, f"unknown ({e})"


def internet():
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
        return True
    except OSError:
        return False


WRITE_PROBE = "preflight/last_check.json"


def hf_check(remote_spec, write_test=True):
    """Returns (ok, user, message).

    Checks, in order: the secret exists; Hugging Face accepts it; it is not a read-only token; it can open the
    repo; and (write_test) it can really WRITE to the repo, by uploading a tiny file (preflight/last_check.json,
    overwritten each time). The write test is the only reliable check: fine-grained tokens are not labelled
    "read" or "write", and a token can be valid yet lack write access to this particular repo.
    """
    if not remote_spec or not remote_spec.startswith("hf:"):
        return True, "-", f"remote is {remote_spec or 'not set'} (no Hugging Face check)"
    repo = remote_spec[3:]
    token = os.environ.get("HF_TOKEN")
    if not token:
        return False, "-", ("HF_TOKEN is not set. On Kaggle: Add-ons -> Secrets -> add HF_TOKEN, and tick it for "
                            "THIS notebook (each imported notebook needs the tick).")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    try:
        info = api.whoami()
    except Exception as e:
        return False, "-", f"HF_TOKEN was rejected ({type(e).__name__}). It may be deleted or mistyped: create a new one."
    user = info.get("name", "?")
    tok = (info.get("auth") or {}).get("accessToken") or {}
    role = tok.get("role", "unknown")
    if role == "read":
        return False, user, ("HF_TOKEN is READ-only. On huggingface.co: Settings -> Access Tokens -> New token, "
                             "type 'Write'; then replace the HF_TOKEN secret in Kaggle.")
    owner = repo.split("/")[0]
    orgs = {o.get("name") for o in info.get("orgs", [])}
    repo_exists = True
    try:
        api.repo_info(repo, repo_type="dataset")
    except Exception as e:
        name = type(e).__name__
        if "NotFound" in name and (owner == user or owner in orgs):
            repo_exists = False
        else:
            return False, user, (f"token of '{user}' cannot open private repo {repo} ({name}). Every Kaggle account "
                                 f"must use a WRITE token from the '{owner}' Hugging Face account (or a member of that org).")
    if not write_test:
        return True, user, f"can access {repo} (token role: {role}; write not tested)"
    try:
        if not repo_exists:
            api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        import datetime
        import json
        import socket
        probe = json.dumps({"checked": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                            "host": socket.gethostname(), "token_role": role}).encode()
        api.upload_file(path_or_fileobj=probe, path_in_repo=WRITE_PROBE, repo_id=repo, repo_type="dataset",
                        commit_message="preflight write check")
    except Exception as e:
        return False, user, (f"token of '{user}' can read but NOT WRITE to {repo} ({type(e).__name__}). "
                             f"Use a token of type 'Write' (or a fine-grained token with write access to this repo) "
                             f"from the '{owner}' account, and update the Kaggle secret.")
    created = " (repo created)" if not repo_exists else ""
    return True, user, f"read + write to {repo} OK{created} (token role: {role})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--remote", default=os.environ.get("PEDGER_REMOTE"))
    p.add_argument("--need-gpu", action="store_true")
    p.add_argument("--no-write-test", action="store_true", help="skip the upload that proves write access")
    args = p.parse_args()
    n_gpu, gpu_names = gpus()
    net = internet()
    ok_hf, user, hf_msg = hf_check(args.remote, not args.no_write_test) if net else (False, "-", "no internet")
    rows = [("session", kaggle_run_type()), ("GPUs", f"{n_gpu}: {gpu_names}"),
            ("internet", "on" if net else "OFF"), ("Hugging Face user", user), ("remote", hf_msg)]
    print("=" * 72)
    for k, v in rows:
        print(f"{k:>18s} | {v}")
    print("=" * 72, flush=True)
    problems = []
    if not net:
        problems.append("Internet is off: notebook Settings -> Internet -> On (needs a phone-verified account).")
    if not ok_hf:
        problems.append(hf_msg)
    if args.need_gpu and n_gpu == 0:
        problems.append("No GPU: notebook Settings -> Accelerator -> GPU T4 x2 (or x1).")
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Interactive":
        print("NOTE: this is an interactive session. For unattended runs use Save Version -> Save & Run All (Commit).")
    for msg in problems:
        print("PROBLEM:", msg)
    if problems:
        sys.exit(1)
    print("preflight OK", flush=True)


if __name__ == "__main__":
    main()
