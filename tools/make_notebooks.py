"""Generate the Kaggle notebooks in notebooks/ (python tools/make_notebooks.py).

Every notebook is meant to be run with Kaggle's "Save Version -> Save & Run All
(Commit)": it then runs on Kaggle's servers (laptop can be off), top to bottom,
with no clicks needed. All state goes to / comes from the private Hugging Face repo,
so every notebook can be re-run on any Kaggle account and continues where the
previous session stopped.
"""
import json
import re
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks"

CONFIG = """# ==== SETTINGS (nothing here needs to change between Kaggle accounts) ====
HF_REPO = "sujalprasad90/pedger-mtkd-store"   # the ONE private Hugging Face repo shared by every account
ACCOUNT = ""                                  # optional label, e.g. your name; sessions are told apart automatically
REPO_URL = ""                                 # optional GitHub URL of the code ("" = Kaggle dataset or HF copy)
# Once per Kaggle account, and for every imported notebook:
#   Add-ons -> Secrets -> HF_TOKEN = a WRITE token from the Hugging Face account that owns HF_REPO, ticked for this notebook
#   Settings -> Internet: On;  Settings -> Accelerator: as written in the title
# The preflight cell below checks all of this and stops with a plain-language fix if something is missing."""

SH = '''def sh(cmd, cwd=None):
    """Run a shell command and show its output; stop the notebook if it fails (a plain `!cmd` carries on)."""
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for raw in p.stdout:                       # progress bars (\\r updates) are shown as their last state
        print(raw.decode(errors="replace").rstrip("\\r\\n").split("\\r")[-1], flush=True)
    if p.wait():
        raise RuntimeError(f"this step failed (exit code {p.returncode}); its error is printed above:\\n  {cmd}")'''

SETUP = """# ---- setup: secrets, code, dependencies (no edits needed) ----
import glob, os, shutil, subprocess, sys, time
SESSION_START = time.time()
""" + SH + """
os.environ["PEDGER_SESSION_START"] = str(SESSION_START)   # long steps stop and save before Kaggle's 12-hour limit
from kaggle_secrets import UserSecretsClient
try:
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    raise RuntimeError("Secret HF_TOKEN is not available: Add-ons -> Secrets -> add HF_TOKEN and tick it for THIS "
                       "notebook, then run again.") from e
os.environ["PEDGER_REMOTE"] = "hf:" + HF_REPO
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.25,<2.0", "pyyaml"], check=True)
W, REPO, WS = "/kaggle/working", "/kaggle/working/pedger-mtkd", "/kaggle/working/ws"
# Always start from the newest code: an interactive session can still hold an old copy from an earlier run.
os.chdir(W)
shutil.rmtree(REPO, ignore_errors=True)
shutil.rmtree(f"{W}/hf_code", ignore_errors=True)
for _m in [m for m in list(sys.modules) if m == "pedger_kd" or m.startswith("pedger_kd.")]:
    del sys.modules[_m]
attached = sorted(glob.glob("/kaggle/input/*/pedger_kd/__init__.py") + glob.glob("/kaggle/input/*/*/pedger_kd/__init__.py"))
if attached:                                       # code uploaded as a Kaggle Dataset (pedger-mtkd-code)
    shutil.copytree(os.path.dirname(os.path.dirname(attached[0])), REPO)
    CODE_SOURCE = "kaggle dataset"
elif REPO_URL:
    subprocess.run(["git", "clone", "-q", REPO_URL, REPO], check=True)
    CODE_SOURCE = "git"
else:                                              # the copy kept on Hugging Face
    from huggingface_hub import snapshot_download
    snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=["code/*"], local_dir=f"{W}/hf_code",
                      token=os.environ["HF_TOKEN"])
    shutil.copytree(f"{W}/hf_code/code", REPO)
    CODE_SOURCE = "huggingface"
_stamp = os.path.join(REPO, "CODE_STAMP.json")
print("code from:", CODE_SOURCE, "| version:", open(_stamp).read().split('"')[3] if os.path.exists(_stamp) else "unstamped")
os.chdir(REPO)
sys.path.insert(0, REPO)
sh("python tools/fetch_upstream.py{extra}")
# preflight: session type, GPUs, internet, Hugging Face account and repo access
if subprocess.run([sys.executable, "-m", "pedger_kd.preflight"]).returncode:
    raise RuntimeError("Preflight failed: fix the PROBLEM lines printed above, then run the notebook again.")"""

PIP_EVAL = """!pip install -q -r requirements.txt
!python tools/install_pyedgeeval.py   # BSDS benchmark scorer; patched build for Python 3.13+, ~1-2 min"""

BSDS_RAW = """# BSDS500 original release (per-annotator ground truth); ~70 MB
RAW_BSDS = "/kaggle/working/BSR/BSDS500/data"
if not os.path.exists(RAW_BSDS):
    !wget -q https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz -O /kaggle/working/bsr.tgz
    !tar -xzf /kaggle/working/bsr.tgz -C /kaggle/working && rm /kaggle/working/bsr.tgz
print(sorted(os.listdir(RAW_BSDS + "/images")))"""

STATUS = """!python -m pedger_kd.sync status"""

CLEANUP = """# keep the notebook output small (everything important is already on Hugging Face)
for d in ["ws/data", "ws/teachers", "ws/.staging", "hf_code", "work", "BSR"]:
    shutil.rmtree(f"/kaggle/working/{d}", ignore_errors=True)
print(f"session used {(time.time() - SESSION_START) / 3600:.2f} h")"""


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    # IPython shell escapes (!cmd) do not reliably support backslash continuation: join those lines
    s = re.sub(r"\\\n\s*", " ", s.strip("\n"))
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def notebook(cells, accel):
    acc = {"gpu": "nvidiaTeslaT4", "cpu": "none"}[accel]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": acc, "isInternetEnabled": True, "isGpuEnabled": accel == "gpu"}}}


def head(title, accel, body):
    return md(f"# {title}\n**Accelerator: {accel}.** Run with **Save Version → Save & Run All (Commit)**; "
              f"you can close the laptop. Re-running continues where the last run stopped, on any account.\n\n{body}")


NOTEBOOKS = {
    "01_data_and_eval_setup.ipynb": ("cpu", [
        head("01 · Data store + evaluation setup  (steps 1-3)", "None (CPU)", """
* builds the BSDS500 array store (agreement maps from the per-annotator ground truth) and uploads it to HF
* copies the code to HF so other accounts don't need the Kaggle code dataset
* **step 1 gate:** scores the released PEdger checkpoint in every inference mode; one of them should
  reproduce the published ODS within ±0.002 (.821/.840 BSDS+VOC or .813/.834 BSDS-only).
  Scoring takes ~4 h per mode on CPU, so this may need two or three runs of this notebook: each run
  continues where the previous one stopped (progress is kept on Hugging Face).
"""),
        code(CONFIG),
        code(SETUP.replace("{extra}", "")),
        code(PIP_EVAL),
        code("""# share the code with every account through Hugging Face: uploads ONLY if this copy is newer than the one
# already there (e.g. a newer attached code dataset); an older copy is never pushed over a newer one
!python -m pedger_kd.sync push-code"""),
        code(BSDS_RAW),
        code("""from pedger_kd.sync import open_remote
remote = open_remote(os.environ["PEDGER_REMOTE"])
if remote.exists("data/bsds/meta.json"):
    print("data/bsds already on HF, pulling it")
    remote.download("data/bsds", WS)
else:
    !python data/prepare_bsds.py --bsds-root {RAW_BSDS} --out {WS}/data/bsds
    !python -m pedger_kd.sync push-dir --local {WS}/data/bsds --to data/bsds
!cat {WS}/data/bsds/meta.json"""),
        code("""# step 3 visual check: image | annotator fraction a | label Y | trust alpha
!python data/visualize_samples.py --store {WS}/data/bsds --split train --n 10 --out {W}/checks/agreement.png
from IPython.display import Image, display; display(Image(f"{W}/checks/agreement.png"))"""),
        code("""# step 1 gate: score the released PEdger checkpoint (single-scale for every mode first; multi-scale if MS=True).
# Scoring is slow on CPU (~4 h per mode for the 200 test images). Progress is saved to Hugging Face every
# 10 minutes and the run stops before Kaggle's 12-hour limit: just run the notebook again to continue.
# Modes already scored (in any session or account) are skipped. The table at the end compares with the paper.
MODES = ["ns_m", "rec_m", "fuse_m"]
MS = False
_modes, _ms = " ".join(MODES), ("--ms" if MS else "")
!python eval/released_gate.py --store {WS}/data/bsds --raw-root {RAW_BSDS} --modes {_modes} {_ms} --out {W}/released --device cpu"""),
        code("""# step 9 part 1: numeric BatchNorm check on the released weights
!python eval/bn_check.py --numeric --ckpt third_party/PEdger/checkpoint_PEdger.pth --device cpu"""),
        code(CLEANUP),
    ]),
    "02_more_datasets.ipynb": ("cpu", [
        head("02 · NYUDv2 and MultiCue stores  (step 12 preparation)", "None (CPU)", """
Attach the downloaded datasets, **inspect their layout first**, write list files `image gt1 [gt2 ...]`
(relative to the raw root), then uncomment the commands. Each store is uploaded to HF.
"""),
        code(CONFIG),
        code(SETUP.replace("{extra}", "")),
        code("""RAW_NYUD = "/kaggle/input/nyud-hed"            # attach and inspect
!ls {RAW_NYUD} | head
# !python data/prepare_list.py --dataset nyud --raw-root {RAW_NYUD} --list {RAW_NYUD}/train_pair.lst --split trainval --out {WS}/data/nyud --max-dist 0.011
# !python data/prepare_list.py --dataset nyud --raw-root {RAW_NYUD} --list {RAW_NYUD}/test.lst --split test --eval-only --out {WS}/data/nyud --max-dist 0.011
# !python data/prepare_list.py --dataset nyud --raw-root {RAW_NYUD} --list my_val.lst --split val --eval-only --out {WS}/data/nyud --max-dist 0.011
# !python -m pedger_kd.sync push-dir --local {WS}/data/nyud --to data/nyud"""),
        code("""RAW_MC = "/kaggle/input/multicue"               # attach; check annotations per image
# !python data/make_multicue_splits.py --list multicue_boundary_all.lst --out {W}/lists/multicue_boundary
# for i in range(3):
#     !python data/prepare_list.py --dataset multicue_boundary --raw-root {RAW_MC} --list {W}/lists/multicue_boundary/split{i}_train.lst --split train --out {WS}/data/multicue_boundary_split{i}
#     !python data/prepare_list.py --dataset multicue_boundary --raw-root {RAW_MC} --list {W}/lists/multicue_boundary/split{i}_test.lst --split test --eval-only --out {WS}/data/multicue_boundary_split{i}
#     !python -m pedger_kd.sync push-dir --local {WS}/data/multicue_boundary_split{i} --to data/multicue_boundary_split{i}"""),
    ]),
    "03_teacher_cache.ipynb": ("gpu", [
        head("03 · Teacher cache  (step 4)", "GPU T4 x1", """
Runs SAUGE, MuGE and MEMO once on the 300 BSDS train+val images (plus a flipped pass that softens the
in-sample maps), packs them, builds the ensembles, and uploads each teacher to HF **as soon as it is done**.
Teachers already on HF are skipped, so if the session ends early just run it again (any account).
If a step fails, the run stops right there with the failing command and its error just above it.
The GPU-hours printed at the end count towards the method's training cost.
"""),
        code(CONFIG),
        code(SETUP.replace("{extra}", " --teachers")),
        code("""from pedger_kd.sync import open_remote
remote = open_remote(os.environ["PEDGER_REMOTE"])
remote.download("data/bsds", WS)
remote.download("teachers/bsds", WS)
SPLIT, TW, TC = "trainval", f"{W}/work", f"{WS}/teachers/bsds"
def on_hf(t): return remote.exists(f"teachers/bsds/{SPLIT}/{t}.npy")
def push(t):
    remote.upload({f"teachers/bsds/{SPLIT}/{t}.npy": f"{TC}/{SPLIT}/{t}.npy",
                   f"teachers/bsds/{SPLIT}/{t}.json": f"{TC}/{SPLIT}/{t}.json"}, f"teacher {t}")
todo = [t for t in ["sauge", "muge", "memo"] if not on_hf(t)]
print("teachers already on HF:", [t for t in ["sauge", "muge", "memo"] if t not in todo], "| still to run:", todo)
if todo:
    sh(f"python teachers/export_images.py --store {WS}/data/bsds --split {SPLIT} --out {TW}/img --flip")"""),
        code("""if "sauge" in todo:
    sh("pip install -q einops")
    sh(f"wget -q -nc https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P {W}")
    for sfx in ["", "_flip"]:
        sh(f"python teachers/run_sauge.py --images {TW}/img{sfx} --out {TW}/sauge{sfx} --sam-ckpt {W}/sam_vit_b_01ec64.pth")
    sh(f"python teachers/pack_cache.py --store {WS}/data/bsds --split {SPLIT} --teacher sauge --pred-dir {TW}/sauge --pred-dir-flip {TW}/sauge_flip --out {TC}")
    push("sauge")"""),
        code("""if "muge" in todo:
    sh("pip install -q efficientnet_pytorch gdown")
    sh(f"gdown -q 15NucsEeHAFwo5O2s11pMR1BikoUtiuUX -O {W}/muge_bsds.pth")
    for sfx in ["", "_flip"]:
        sh(f"python teachers/run_muge.py --images {TW}/img{sfx} --out {TW}/muge{sfx} --ckpt {W}/muge_bsds.pth")
    sh(f"python teachers/pack_cache.py --store {WS}/data/bsds --split {SPLIT} --teacher muge --pred-dir {TW}/muge --pred-dir-flip {TW}/muge_flip --out {TC}")
    push("muge")"""),
        code("""if "memo" in todo:
    from importlib.metadata import version
    MEMO = f"{REPO}/third_party/MEMO_Edge_Detection"
    # Only what MEMO's prediction code imports. Its requirements.txt is unpinned, would replace Kaggle's torch, lists
    # packages inference never uses (deepspeed, tensorrt, onnxruntime, gradio) and omits peft, which it does import.
    open(f"{W}/keep_torch.txt", "w").write(f"torch=={version('torch')}\\ntorchvision=={version('torchvision')}\\n")
    sh(f"pip install -q pytorch_lightning omegaconf diffusers einops peft -c {W}/keep_torch.txt")
    sh("python download_checkpoints.py --model bsds-late --model synthetic-late", cwd=MEMO)
    for sfx in ["", "_flip"]:
        sh(f"python edge_prediction.py --test_folder {TW}/img{sfx} --save_folder {TW}/memo{sfx} --config_file configs/discrete_BSDS_finetune/binary_lora_default.yaml --model_path pretrained_models/MEMO_BSDS_ft_late/mp_rank_00_model_states.pt --base_model_path pretrained_models/MEMO_synthetic_late/mp_rank_00_model_states.pt --guidance_scale 1.4 --max_steps 20", cwd=MEMO)
    sh(f"python teachers/pack_cache.py --store {WS}/data/bsds --split {SPLIT} --teacher memo --pred-dir {TW}/memo/prediction --pred-dir-flip {TW}/memo_flip/prediction --out {TC}")
    push("memo")"""),
        code("""# ensembles (trainval) + the same maps cut out for the dev split (train) used by the ablations
remote.download("teachers/bsds", WS)
ALL = "sauge muge memo ensemble_mean ensemble_confidence"
for mode in ["mean", "confidence"]:
    sh(f"python teachers/ensemble.py --teacher-root {TC} --split {SPLIT} --teachers sauge muge memo --mode {mode}")
sh(f"python teachers/subset_cache.py --store {WS}/data/bsds --teacher-root {TC} --from trainval --to train --teachers {ALL}")
sh(f"python -m pedger_kd.sync push-dir --local {TC} --to teachers/bsds")
print(f"teacher caching this session: {(time.time() - SESSION_START) / 3600:.2f} GPU-h")"""),
        code("""# step 6 gate: red = teacher, green = label after the joint augmentation; they must trace the same contours
sh(f"python data/visualize_samples.py --store {WS}/data/bsds --split {SPLIT} --n 8 --teacher-root {TC} --teacher ensemble_mean --augmented --out {W}/checks/aug_align.png")
from IPython.display import Image, display; display(Image(f"{W}/checks/aug_align.png"))"""),
        code(CLEANUP),
    ]),
    "04_timing_and_checks.ipynb": ("gpu", [
        head("04 · Timing, BatchNorm check, baseline  (steps 8-9)", "GPU T4 x2", """
1. **Timing** (5 min): estimated hours per full run → plan the rest from it (uploaded to `timing/`).
2. **Queue `queues/checks.yaml`:** BatchNorm-check runs (micro-batch 1 / 4 / 4-with-stock-BN) and the
   baseline seed 0. Both T4s train different jobs. Re-run until `status` shows all four `done`.
Scores for the checks are computed by notebook 06 (CPU).
"""),
        code(CONFIG),
        code(SETUP.replace("{extra}", "")),
        code("""from pedger_kd.sync import open_remote
remote = open_remote(os.environ["PEDGER_REMOTE"])
remote.download("data/bsds", WS)
if not remote.exists("timing/baseline.json"):
    !python bench/time_run.py --config configs/baseline.yaml --data-root {WS}/data --out {W}/timing/baseline.json
    !python -m pedger_kd.sync push-dir --local {W}/timing --to timing
else:
    remote.download("timing", W)
!cat {W}/timing/baseline.json"""),
        code("""HOURS = 11.25            # whole-session budget (Kaggle stops GPU sessions at 12 h)
!python -m pedger_kd.jobs --queue queues/checks.yaml --ws {WS} --hours {HOURS} --session-start {SESSION_START} --gpus auto --label '{ACCOUNT}'"""),
        code(STATUS),
        code(CLEANUP),
    ]),
    "05_train_queue.ipynb": ("gpu", [
        head("05 · Train the queue  (steps 10-12)", "GPU T4 x2", """
Set `QUEUE` and run. Each session trains for ~11 h, saving a checkpoint every 30 min and at every epoch, and
uploads to HF in the background. A half-finished job continues exactly where it stopped. Two T4s train two
different jobs. When this account's GPU quota is used up, run the same notebook on another account (same
`HF_REPO`, its own `HF_TOKEN` secret): it picks up the queue automatically.

Queues: `queues/main.yaml` (main rows, seeds 0→1→2), `queues/ablations.yaml`, `queues/nyud.yaml`,
`queues/multicue.yaml`.
"""),
        code(CONFIG),
        code("""QUEUE = "queues/main.yaml"
HOURS = 11.25            # whole-session budget, counted from the first cell (Kaggle limit: 12 h)"""),
        code(SETUP.replace("{extra}", "")),
        code(STATUS),
        code("""!python -m pedger_kd.jobs --queue {QUEUE} --ws {WS} --hours {HOURS} --session-start {SESSION_START} --gpus auto --label '{ACCOUNT}' --squash"""),
        code(STATUS),
        code(CLEANUP),
    ]),
    "06_evaluate.ipynb": ("cpu", [
        head("06 · Evaluate and analyse  (steps 9, 10, 13, 14)", "None (CPU, uses no GPU quota)", """
Scores every finished run that has no scores yet (dev/ablation runs on val, final runs on test), uploads the
scores, builds the tables, the BatchNorm-check comparison, the agreement-bin analysis and the efficiency
table. Safe to re-run at any time.
"""),
        code(CONFIG),
        code(SETUP.replace("{extra}", "")),
        code(PIP_EVAL),
        code(BSDS_RAW),
        code("""!python eval/eval_runs.py --remote hf:{HF_REPO} --ws {WS} --raw-root bsds={RAW_BSDS} --nproc 4
# multi-scale for the main rows only:
# !python eval/eval_runs.py --remote hf:{HF_REPO} --ws {WS} --raw-root bsds={RAW_BSDS} --nproc 4 --ms --runs "*_s?" """),
        code("""!python results/make_tables.py --scores "{WS}/runs/*/scores_*.json" --out {W}/results/tables.md
!python -m pedger_kd.sync push-dir --local {W}/results --to results"""),
        code("""# step 9: BatchNorm check (needs the three bn_* runs scored on val)
r = f"{WS}/runs"
if all(os.path.exists(f"{r}/{c}_s0/scores_val_ns_m.json") for c in ["bn_micro1", "bn_micro4", "bn_stock4"]):
    !python eval/bn_check.py --compare-runs {r}/bn_micro1_s0/scores_val_ns_m.json {r}/bn_micro4_s0/scores_val_ns_m.json {r}/bn_stock4_s0/scores_val_ns_m.json
else:
    print("BatchNorm-check runs not finished/scored yet")"""),
        code("""# step 13: agreement bins (test split, seed 0)
runs = {"PEdger": "baseline_s0", "A_uniform": "A_uniform_s0", "Ours": "A_agree_s0"}
args = " ".join(f"--run {k}={WS}/runs/{v}/scores_test_ns_m.json" for k, v in runs.items()
                if os.path.exists(f"{WS}/runs/{v}/scores_test_ns_m.json"))
if args.count("--run") >= 2:
    from pedger_kd.sync import open_remote
    remote = open_remote(os.environ["PEDGER_REMOTE"])
    for v in runs.values():
        remote.download(f"runs/{v}/pred_test_ns_m", WS)
    !python -m pedger_kd.sync pull --prefix data/bsds/eval_lists --ws {WS}
    !python eval/agreement_bins.py --store {WS}/data/bsds --split test --raw-root {RAW_BSDS} {args} --out {W}/results/agreement_bins.json
    !python -m pedger_kd.sync push-dir --local {W}/results --to results
else:
    print("need at least two scored test runs")"""),
        code("""# step 14: efficiency + training cost (CPU FPS here; run bench/speed_flops.py in a GPU notebook for T4 FPS)
TEACHER_HOURS = 0.0   # GPU-hours printed by notebook 03
!python bench/speed_flops.py --size 320 480 --runs "{WS}/runs/*" --teacher-hours {TEACHER_HOURS} --out {W}/results/efficiency.json
!python -m pedger_kd.sync push-dir --local {W}/results --to results"""),
        code(CLEANUP),
    ]),
}


def main():
    NB.mkdir(exist_ok=True)
    for old in NB.glob("*.ipynb"):
        old.unlink()
    for name, (accel, cells) in NOTEBOOKS.items():
        (NB / name).write_text(json.dumps(notebook(cells, accel), indent=1))
    print(f"wrote {len(NOTEBOOKS)} notebooks to {NB}")


if __name__ == "__main__":
    main()
