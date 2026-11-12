"""Generate the Kaggle notebooks in notebooks/ (python tools/make_notebooks.py)."""
import json
import re
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = """# --- setup: code + upstream PEdger -------------------------------------------
import os, subprocess, sys
REPO_URL = ""          # e.g. "https://github.com/<you>/pedger-mtkd.git" (or attach the code as a Kaggle Dataset)
CODE_DATASET = "/kaggle/input/pedger-mtkd-code"
W = "/kaggle/working"
REPO = f"{W}/pedger-mtkd"
if not os.path.exists(REPO):
    if REPO_URL:
        subprocess.run(["git", "clone", "-q", REPO_URL, REPO], check=True)
    else:
        subprocess.run(["cp", "-r", CODE_DATASET, REPO], check=True)
os.chdir(REPO)
sys.path.insert(0, REPO)
!python tools/fetch_upstream.py{extra}
!pip install -q -r requirements.txt"""

DATA_PATHS = """# --- inputs (attach these Kaggle Datasets; names are suggestions) --------------
RAW_BSDS = "/kaggle/input/bsds500-raw/BSR/BSDS500/data"     # original Berkeley release
DATA = "/kaggle/input/pedger-data"                           # output of 01 (array stores), saved as a Dataset
CACHE = "/kaggle/input/pedger-teachers"                      # output of 03 (teacher cache), saved as a Dataset
RUNS = "/kaggle/working/runs"
import os
if not os.path.exists(DATA): DATA = "/kaggle/working/data_store"
if not os.path.exists(CACHE): CACHE = "/kaggle/working/teacher_cache\""""


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    # IPython shell escapes (!cmd) do not reliably support backslash continuation: join those lines
    s = re.sub(r"\\\n\s*", " ", s.strip("\n"))
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def notebook(cells, gpu):
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4" if gpu else "none", "isInternetEnabled": True}}}


NOTEBOOKS = {
    "01_eval_setup.ipynb": (False, [
        md("""# 01 · Evaluation setup + data store  (steps 1-3, CPU session)
**Gate (step 1):** the released PEdger checkpoint must reproduce its published ODS within ±0.002
with our Python benchmark before any number from this pipeline is trusted.
The released checkpoint was saved at epoch 19 and its training set is not stated; compare against
the published BSDS+VOC (.821/.840) and BSDS-only (.813/.834) numbers and note which one it matches.
Output of this notebook (`/kaggle/working/data_store`) → save as Kaggle Dataset **pedger-data**."""),
        code(SETUP.replace("{extra}", "")),
        code("""# BSDS500 original release (per-annotator .mat ground truth)
RAW_BSDS = "/kaggle/working/BSR/BSDS500/data"
if not os.path.exists(RAW_BSDS):
    !wget -q https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz -O /kaggle/working/bsr.tgz
    !tar -xzf /kaggle/working/bsr.tgz -C /kaggle/working && rm /kaggle/working/bsr.tgz
!ls {RAW_BSDS}/images"""),
        md("## Steps 2-3: array store + agreement maps (annotator fraction computed from the per-annotator .mat files)"),
        code("""!python data/prepare_bsds.py --bsds-root {RAW_BSDS} --out /kaggle/working/data_store/bsds
!cat /kaggle/working/data_store/bsds/meta.json"""),
        code("""# Visual check (step 3 gate: maps look right on 10 images): image | a | Y | alpha
!python data/visualize_samples.py --store /kaggle/working/data_store/bsds --split train --n 10 --out /kaggle/working/checks/agreement.png
from IPython.display import Image; Image("/kaggle/working/checks/agreement.png")"""),
        md("""## Step 1: score the released PEdger checkpoint in every inference mode
`ns_m` = non-recurrent momentum net (0.73M, the deployed model), `rec_m` = recurrent, `fuse_m` = both fused.
`--ms` = multi-scale. Pick the mode that reproduces the paper and set `eval.deploy` in `configs/base.yaml`."""),
        code("""CKPT = "third_party/PEdger/checkpoint_PEdger.pth"
for mode in ["ns_m", "rec_m", "fuse_m"]:
    for ms in ["", "--ms"]:
        out = f"/kaggle/working/released/{mode}{'_ms' if ms else ''}"
        !python eval/run_bsds_eval.py --ckpt {CKPT} --mode {mode} {ms} --store /kaggle/working/data_store/bsds \\
            --split test --raw-root {RAW_BSDS} --out-dir {out} --nproc 4"""),
        code("""import glob, json
for f in sorted(glob.glob("/kaggle/working/released/*/scores_*.json")):
    if f.endswith("_per_image.json"): continue
    r = json.load(open(f)); print(f"{f.split('/')[-2]:12s} ODS {r['ODS']:.4f}  OIS {r['OIS']:.4f}  AP {r['AP']:.4f}")
print("published: .821/.840 (BSDS+VOC)  .813/.834 (BSDS only)")"""),
        code("""# Numeric BatchNorm check on the released weights (step 9, part 1)
!python eval/bn_check.py --numeric --ckpt third_party/PEdger/checkpoint_PEdger.pth --device cpu"""),
    ]),
    "02_prepare_data.ipynb": (False, [
        md("""# 02 · NYUDv2 and MultiCue stores  (step 12 preparation, CPU session)
Verify the annotation format of each download first (plan: *verify when downloading*), then write
list files `image gt1 [gt2 ...]` (paths relative to the raw root). Append to Dataset **pedger-data**."""),
        code(SETUP.replace("{extra}", "")),
        code("""# NYUDv2 (HED/RCF version): one annotation per image, maxDist 0.011
RAW_NYUD = "/kaggle/input/nyud-hed"   # attach; inspect its layout first
!ls {RAW_NYUD} | head
# !python data/prepare_list.py --dataset nyud --raw-root {RAW_NYUD} --list {RAW_NYUD}/train_pair.lst --split trainval --out /kaggle/working/data_store/nyud --max-dist 0.011
# !python data/prepare_list.py --dataset nyud --raw-root {RAW_NYUD} --list {RAW_NYUD}/test_pair.lst --split test --eval-only --out /kaggle/working/data_store/nyud --max-dist 0.011
# For the teacher gate, also hold out a val list from train (e.g. 100 images) and prepare it with --eval-only --split val"""),
        code("""# MultiCue boundaries: 3 random 80/20 scene splits (fixed seeds 0, 1, 2)
RAW_MC = "/kaggle/input/multicue"   # attach; check the number of annotations per image
# !python data/make_multicue_splits.py --list multicue_boundary_all.lst --out /kaggle/working/lists/multicue_boundary
# for i in range(3):
#     !python data/prepare_list.py --dataset multicue_boundary --raw-root {RAW_MC} --list /kaggle/working/lists/multicue_boundary/split{i}_train.lst --split train --out /kaggle/working/data_store/multicue_boundary_split{i}
#     !python data/prepare_list.py --dataset multicue_boundary --raw-root {RAW_MC} --list /kaggle/working/lists/multicue_boundary/split{i}_test.lst --split test --eval-only --out /kaggle/working/data_store/multicue_boundary_split{i}"""),
    ]),
    "03_cache_teachers.ipynb": (True, [
        md("""# 03 · Teacher caching  (step 4, GPU session, run once)
Each teacher runs once on the training images (and on validation images for the gate); outputs are
packed into one aligned uint8 cache. Save `/kaggle/working/teacher_cache` as Dataset **pedger-teachers**.
Log the GPU-hours of this notebook: they count towards the method's training cost.
BSDS teachers are trained on BSDS train+val, so their maps on these images are *in-sample*
(sharper than on unseen images). `--flip` averages with a flipped pass to soften them (optional)."""),
        code(SETUP.replace("{extra}", " --teachers")),
        code(DATA_PATHS),
        code("""import time; T0 = time.time()
SPLIT = "trainval"           # use "train" too if you run dev-split ablations
!python teachers/export_images.py --store {DATA}/bsds --split {SPLIT} --out /kaggle/working/work/img_{SPLIT} --flip"""),
        code("""# SAUGE (SAM ViT-B frozen + side network; weights in its repo)
!pip install -q einops
!wget -q -nc https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P /kaggle/working
for suffix in ["", "_flip"]:
    !python teachers/run_sauge.py --images /kaggle/working/work/img_{SPLIT}{suffix} --out /kaggle/working/work/sauge_{SPLIT}{suffix} --sam-ckpt /kaggle/working/sam_vit_b_01ec64.pth"""),
        code("""# MuGE (weights on Google Drive)
!pip install -q efficientnet_pytorch gdown
!gdown -q 15NucsEeHAFwo5O2s11pMR1BikoUtiuUX -O /kaggle/working/muge_bsds.pth
for suffix in ["", "_flip"]:
    !python teachers/run_muge.py --images /kaggle/working/work/img_{SPLIT}{suffix} --out /kaggle/working/work/muge_{SPLIT}{suffix} --ckpt /kaggle/working/muge_bsds.pth"""),
        code("""# MEMO (BSDS LoRA weights + synthetic base model on Hugging Face), via its own CLI
%cd third_party/MEMO_Edge_Detection
!pip install -q -r requirements.txt
!python download_checkpoints.py --model bsds-late --model synthetic-late
for suffix in ["", "_flip"]:
    !python edge_prediction.py --test_folder /kaggle/working/work/img_{SPLIT}{suffix} --save_folder /kaggle/working/work/memo_{SPLIT}{suffix} \\
        --config_file configs/discrete_BSDS_finetune/binary_lora_default.yaml \\
        --model_path pretrained_models/MEMO_BSDS_ft_late/mp_rank_00_model_states.pt \\
        --base_model_path pretrained_models/MEMO_synthetic_late/mp_rank_00_model_states.pt --guidance_scale 1.4 --max_steps 20
%cd /kaggle/working/pedger-mtkd"""),
        code("""W = "/kaggle/working/work"
for t, sub in [("sauge", ""), ("muge", ""), ("memo", "/prediction")]:
    !python teachers/pack_cache.py --store {DATA}/bsds --split {SPLIT} --teacher {t} --pred-dir {W}/{t}_{SPLIT}{sub} --pred-dir-flip {W}/{t}_{SPLIT}_flip{sub} --out /kaggle/working/teacher_cache/bsds
for mode in ["mean", "confidence"]:
    !python teachers/ensemble.py --teacher-root /kaggle/working/teacher_cache/bsds --split {SPLIT} --teachers sauge muge memo --mode {mode}
print(f"teacher caching GPU-hours: {(time.time() - T0) / 3600:.2f}")"""),
        code("""# Alignment check (step 6 gate): red = teacher, green = label; they must trace the same contours
!python data/visualize_samples.py --store {DATA}/bsds --split {SPLIT} --n 8 --teacher-root /kaggle/working/teacher_cache/bsds --teacher ensemble_mean --augmented --out /kaggle/working/checks/aug_align.png
from IPython.display import Image; Image("/kaggle/working/checks/aug_align.png")"""),
        md("""## NYUDv2 / MultiCue: teacher gate
Run the teachers on the dataset's training images (pack as above, with `--fit pad32` for SAUGE/MuGE)
and on its validation images (`export_images.py --eval-split val`), then:
```
!python teachers/teacher_gate.py --store {DATA}/nyud --split val --raw-root RAW_NYUD \\
    --teacher sauge=.../val_sauge --teacher muge=.../val_muge --teacher memo=.../val_memo/prediction \\
    --teacher ensemble=.../val_ensemble --baseline-scores runs/nyud_baseline_s0/scores_val_ns_m.json \\
    --out /kaggle/working/teacher_cache/nyud/gate.json
```
Rebuild `ensemble_mean` from the kept teachers only."""),
    ]),
    "04_baseline_and_checks.ipynb": (True, [
        md("""# 04 · Timing, BatchNorm check, baseline  (steps 8-9, GPU session)
1. **Time first.** Everything else is planned from this number.
2. **BatchNorm check:** three short runs; micro-batch 4 must score like micro-batch 1.
3. **Baseline reproduction:** official PEdger in our pipeline (gate: within ~0.005 of the published
   number, otherwise our retrain is the declared baseline)."""),
        code(SETUP.replace("{extra}", "")),
        code(DATA_PATHS),
        code("""!python bench/time_run.py --config configs/baseline.yaml --data-root {DATA} --teacher-root {CACHE} --out /kaggle/working/timing_baseline.json
!python bench/time_run.py --config configs/A_agree.yaml --data-root {DATA} --teacher-root {CACHE} --out /kaggle/working/timing_A_agree.json
# If peak memory allows, try a larger micro-batch (keep micro_batch x accum_steps = 16):
# !python bench/time_run.py --config configs/baseline.yaml --data-root {DATA} --teacher-root {CACHE} --set train.micro_batch=8 train.accum_steps=2"""),
        md("""**Read the estimate.** `est_hours_full_run` x number of runs vs ~30 GPU-h/week. If one full run does
not fit the budget, shorten the schedule for **every** row in the same way (e.g. `--set data.samples_per_epoch=...`),
state it in the paper, and compare only against our own retrained baseline in that schedule."""),
        code("""# BatchNorm check (step 9): short runs on the dev split, scored on val
for c in ["bn_micro1", "bn_micro4", "bn_stock4"]:
    !python -m pedger_kd.train_kd --config configs/checks/{c}.yaml --data-root {DATA} --teacher-root {CACHE} --out {RUNS}/{c} --max-hours 3.5"""),
        code("""for c in ["bn_micro1", "bn_micro4", "bn_stock4"]:
    !python eval/run_bsds_eval.py --run {RUNS}/{c} --store {DATA}/bsds --split val --raw-root {RAW_BSDS} --nproc 4
!python eval/bn_check.py --compare-runs {RUNS}/bn_micro1/scores_val_ns_m.json {RUNS}/bn_micro4/scores_val_ns_m.json {RUNS}/bn_stock4/scores_val_ns_m.json"""),
        code("""# Baseline seed 0 (resumable: re-run the same cell in a new session; attach the previous output as a Dataset
# and copy runs/ back into /kaggle/working first)
!python -m pedger_kd.train_kd --config configs/baseline.yaml --seed 0 --data-root {DATA} --teacher-root {CACHE} --out {RUNS}/baseline_s0 --max-hours 11.5"""),
    ]),
    "05_train.ipynb": (True, [
        md("""# 05 · Train one row  (step 10-12, GPU session)
Pick a config and seed. Checkpoints every epoch; `--max-hours` stops cleanly before Kaggle's 12 h limit and
the same command resumes. Main rows: baseline, B_uniform, A_uniform, A_agree × seeds 0, 1, 2 (minimum 2).
Ablations: `configs/ablations/*.yaml`, seed 0. Two T4s: run two configs at once with
`CUDA_VISIBLE_DEVICES=0` / `=1` in two background processes."""),
        code(SETUP.replace("{extra}", "")),
        code(DATA_PATHS),
        code("""CONFIG, SEED = "configs/A_agree.yaml", 0
NAME = os.path.splitext(os.path.basename(CONFIG))[0]
PREV = "/kaggle/input/pedger-runs"          # optional: previous session's runs/ to resume from
if os.path.exists(f"{PREV}/{NAME}_s{SEED}"):
    !mkdir -p {RUNS} && cp -r {PREV}/{NAME}_s{SEED} {RUNS}/
!python -m pedger_kd.train_kd --config {CONFIG} --seed {SEED} --data-root {DATA} --teacher-root {CACHE} --out {RUNS}/{NAME}_s{SEED} --max-hours 11.5"""),
        code("""# Optional: two runs in parallel on Kaggle's 2xT4
# !CUDA_VISIBLE_DEVICES=0 nohup python -m pedger_kd.train_kd --config configs/baseline.yaml --seed 1 --data-root {DATA} --teacher-root {CACHE} --out {RUNS}/baseline_s1 --max-hours 11.5 > {RUNS}/baseline_s1.log 2>&1 &
# !CUDA_VISIBLE_DEVICES=1 nohup python -m pedger_kd.train_kd --config configs/A_agree.yaml --seed 1 --data-root {DATA} --teacher-root {CACHE} --out {RUNS}/A_agree_s1 --max-hours 11.5 > {RUNS}/A_agree_s1.log 2>&1 &"""),
    ]),
    "06_evaluate.ipynb": (False, [
        md("""# 06 · Evaluate and analyse  (steps 10, 13, 14; CPU session, no GPU quota)
Attach the finished runs (Dataset **pedger-runs**). Final numbers: test split, models trained on trainval.
Ablations / dev runs: val split."""),
        code(SETUP.replace("{extra}", "")),
        code(DATA_PATHS + """
RUNS = "/kaggle/input/pedger-runs"
OUT = "/kaggle/working/eval_runs"
!mkdir -p {OUT} && cp -r {RUNS}/* {OUT}/"""),
        code("""import glob, json
for run in sorted(glob.glob(f"{OUT}/*")):
    if not os.path.exists(f"{run}/config.yaml"): continue
    split = "val" if "split: train\\n" in open(f"{run}/config.yaml").read() else "test"
    !python eval/run_bsds_eval.py --run {run} --store {DATA}/bsds --split {split} --raw-root {RAW_BSDS} --nproc 4 --device cpu
    # multi-scale for the main rows only:
    # !python eval/run_bsds_eval.py --run {run} --ms --store {DATA}/bsds --split {split} --raw-root {RAW_BSDS} --nproc 4 --device cpu"""),
        code("""!python results/make_tables.py --scores "{OUT}/*/scores_*.json" --out /kaggle/working/tables.md"""),
        code("""# Agreement-bin analysis (step 13): baseline vs ours, seed 0, test split
!python eval/agreement_bins.py --store {DATA}/bsds --split test --raw-root {RAW_BSDS} \\
    --run PEdger={OUT}/baseline_s0/scores_test_ns_m.json --run Ours={OUT}/A_agree_s0/scores_test_ns_m.json \\
    --run A_uniform={OUT}/A_uniform_s0/scores_test_ns_m.json --out /kaggle/working/agreement_bins.json"""),
        code("""# Efficiency and training cost (step 14). Run once on a GPU session for T4 FPS; CPU FPS works here.
TEACHER_HOURS = 0.0   # from notebook 03
!python bench/speed_flops.py --size 320 480 --runs "{OUT}/*" --teacher-hours {TEACHER_HOURS} --out /kaggle/working/efficiency.json"""),
    ]),
}


def main():
    NB.mkdir(exist_ok=True)
    for name, (gpu, cells) in NOTEBOOKS.items():
        (NB / name).write_text(json.dumps(notebook(cells, gpu), indent=1))
    print(f"wrote {len(NOTEBOOKS)} notebooks to {NB}")


if __name__ == "__main__":
    main()
