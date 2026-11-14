# pedger-mtkd

Agreement-aware multi-teacher distillation for PEdger: SAUGE, MuGE and MEMO teach the 0.73M-parameter
PEdger edge detector, and per-pixel human agreement on BSDS500 decides where the student follows the
human label and where it follows the models. Inference is unchanged PEdger.

Everything runs on Kaggle (T4). The official PEdger code is used unmodified (`third_party/PEdger`, fetched
at a pinned commit); this repo adds the data pipeline, the two distillation variants, the checks and the
analysis.

## Facts from the official code that change the plan

| # | Finding | What the repo does |
|---|---|---|
| 1 | The **0.73M** model is the **non-recurrent** network (`Guider_noshare`, 734,225 params). The recurrent one has 0.36M. `test_speed.py` times the non-recurrent one. | Deploy = non-recurrent momentum network (`eval.deploy: ns_m`). Step 1 confirms which output reproduces the paper. |
| 2 | PEdger trains with **batch 1 + 16-step accumulation**, and every BatchNorm has `track_running_stats=False`, so it always uses the current batch's statistics. Batching 4 images with plain BN changes the released model's logits by up to **132**. | Per-image normalisation (same parameters, same checkpoint format): a micro-batch of B is numerically identical to B batch-1 passes (float64 test: 1e-14). The BatchNorm check is still run on real training. |
| 3 | LR schedule: warm-up **1e-3 → 8e-3** over 4 epochs, then cosine (not "1e-3 then linear decay"). | The official scheduler class is reused; a test pins the exact 30-epoch sequence. |
| 4 | The official README says training "lasts about 7 days" (BSDS+VOC, full resolution, batch 1). | **Time before planning** (`bench/time_run.py`). If a full run doesn't fit the GPU budget, shorten the schedule identically for every row and compare only against our own retrained baseline. |
| 5 | MEMO's BSDS weights **are** released (Hugging Face). | MEMO is teacher 3; DiffusionEdge stays an unused backup. |
| 6 | The released checkpoint is from epoch 19, and its training set (BSDS-only or +VOC) is not stated. | Step 1 compares it with both published numbers. |
| 7 | PEdger's data is the pre-augmented HED-BSDS set (16 rotations × 2 flips × 3 scales = 28,800 images). | The same augmentation family is applied on the GPU to the 300 originals, jointly to the image, label, teacher map and agreement map. One epoch = 300 × 96 samples, as in HED-BSDS. Rotations crop to the valid region (no black borders). Teachers run once on 300 images instead of 28,800. |

## Flow of work

```
SETUP & TRUST           BUILD                 VERIFY                    RUN                  ANALYZE
1 Eval setup      ->    5 Joint GPU aug  ->   7 Sanity tests     ->     10 Main runs    ->   13 Agreement bins
2 Data prep             6 Variants A + B      8 Baseline reproduction   11 Ablations         14 Efficiency table
3 Agreement maps                              9 BatchNorm check         12 NYUDv2 + MultiCue
4 Teacher cache
```

| Step | Command (details in `notebooks/`) | Gate |
|---|---|---|
| 1 Eval setup | `eval/run_bsds_eval.py --ckpt third_party/PEdger/checkpoint_PEdger.pth --mode {ns_m,rec_m,fuse_m} [--ms]` | Released checkpoint reproduces its published ODS within ±0.002 |
| 2 Data prep | `data/prepare_bsds.py`; `data/prepare_list.py` (NYUDv2, MultiCue); `data/make_multicue_splits.py` | Loaders return correct images and labels (`data/visualize_samples.py`) |
| 3 Agreement maps | built by step 2 from the per-annotator `.mat` files (0/1/2 px dilation) | Maps look right on 10 images |
| 4 Teacher cache | `teachers/export_images.py` → `run_sauge.py`, `run_muge.py`, MEMO CLI → `pack_cache.py` → `ensemble.py`; `teacher_gate.py` (NYUD/MultiCue) | Teacher table; teachers not better than PEdger dropped per dataset |
| 5 Joint augmentation | `pedger_kd/augment.py` | `tests/test_augment_align.py` + overlay `visualize_samples.py --augmented` |
| 6 Variants A + B | `pedger_kd/soft_target.py`, `pedger_kd/kd_loss.py`, `pedger_kd/train_kd.py` | — |
| 7 **Sanity tests** | `pytest tests` | λ_kd = 0 ≡ baseline; γ = 0, α = 1 − η ≡ PEdger target (both through the real training step) |
| 8 **Baseline** | `bench/time_run.py` then `train_kd --config configs/baseline.yaml` | Within ~0.005 of published, else our retrain is the declared baseline |
| 9 **BatchNorm check** | `configs/checks/bn_*.yaml` + `eval/bn_check.py --compare-runs` | Micro-batch 4 scores like micro-batch 1 |
| 10 Main runs | `baseline`, `B_uniform`, `A_uniform`, `A_agree` × seeds 0-2 (minimum 2) | All runs log GPU-hours |
| 11 Ablations | `configs/ablations/*.yaml`, 1 seed, short schedule, val split | short-vs-full ranking check (`short_check_*` vs `full_check_*`) |
| 12 Other datasets | `configs/nyud/*`, `configs/multicue/*` | — |
| 13 **Agreement bins** | `eval/agreement_bins.py` | Gains concentrated in the low-agreement bin (H2) |
| 14 Efficiency | `bench/speed_flops.py` | Params/FLOPs/FPS identical to PEdger; training hours include teacher caching |

Tables: `results/make_tables.py` groups runs by setting (dataset, split, train split, ±VOC, SS/MS,
schedule). It never mixes settings in one table and shows published numbers (†) only where the setting matches.

## Protocol decisions

- **Main setting: BSDS-only** (no PASCAL VOC); published PEdger .813/.834 in this setting.
- **Development:** train on `train` (200 images), score on `val`. **Final:** train on `trainval` (300), score on `test` once.
- **The BSDS teachers were trained on BSDS train+val,** so their maps on our training images are in-sample. Optionally average them with a flipped pass (`pack_cache.py --pred-dir-flip`) and disclose this. The teacher gate is meaningless on BSDS val and is used only on NYUDv2/MultiCue.
- **Main-row comparison for H2:** `A_uniform` (α = 1 − η everywhere) vs `A_agree` (α from agreement). Same teachers, same everything else.
- With agreement-based α, pixels where **all** annotators agree on "no edge" get α = 1, so the teachers never add edges there. `A_uniform` and Variant B cover that case.

## Running it on Kaggle, step by step

**How unattended training works.** All state lives in one private Hugging Face dataset repo
(`HF_REPO`): data stores, teacher cache, checkpoints, scores, and a copy of the code. Every notebook is run
with **Save Version → Save & Run All (Commit)**, so it runs on Kaggle's servers (the laptop can be off) for
up to 12 h. Training checkpoints every 30 min and at every epoch end, uploads in the background, stops cleanly
before the 12 h limit, and resumes **exactly** where it stopped (tested bit-identical across 5 interruptions).
A queue (`queues/*.yaml`) plus a heartbeat lock lets any session on **any Kaggle account** with the same
`HF_REPO` and an `HF_TOKEN` secret take the next unfinished job. A crash or quota cut-off loses at most
30 min; that job becomes claimable again after 75 min without a heartbeat.

### One-time setup
1. **Hugging Face:** settings → Access Tokens → *New token*, type **Write**. Delete any token that was shared in
   plain text. The private repo `HF_REPO` is created automatically on first use.
2. **Each Kaggle account** (phone-verified, so Internet and GPU are available):
   *Add-ons → Secrets → Add* `HF_TOKEN` = a **Write** token from the Hugging Face account that **owns** `HF_REPO`
   (every Kaggle account uses a token from that same HF account; a teammate's own HF token cannot open the
   private repo). Kaggle's terms allow one account per person, so use accounts that belong to different team members.
3. **Notebooks:** on kaggle.com → *Create → Notebook → File → Import notebook*, one file at a time from
   `notebooks/`. In each: tick the `HF_TOKEN` secret (Add-ons → Secrets), *Settings → Internet: On*,
   accelerator as written in the notebook title. Nothing in the first cell needs to change between accounts;
   `ACCOUNT` is an optional label (sessions are told apart automatically).
   **Preflight:** every notebook starts by checking the session type, GPUs, internet, and the token: it must be
   accepted by Hugging Face, must not be read-only, must open the repo, and must **actually write** to it (it
   uploads a tiny `preflight/last_check.json`). If anything fails, the run stops at once with a plain-language fix.
   Check yourself any time with `python -m pedger_kd.preflight`.
4. **Code:** already on HF under `code/` (the notebooks fetch it from there). After changing code, push it
   again with `python -m pedger_kd.sync --remote hf:<HF_REPO> push-code`, or upload
   `dist/pedger-mtkd-code.zip` (`tools/make_code_zip.py`) as a Kaggle Dataset and attach it. An attached copy
   takes precedence.

### Running order (each = *Save Version → Save & Run All*; check results in the version's log/output)
| # | Notebook | Accelerator | Done when |
|---|---|---|---|
| 1 | `01_data_and_eval_setup` | None | Released-checkpoint table in the log; one mode within ±0.002 of the paper (step 1 gate). Set `eval.deploy` in `configs/base.yaml` if it's not `ns_m`. |
| 2 | `03_teacher_cache` | GPU T4 x1 | Log prints the alignment picture and the caching GPU-hours. If it timed out, run again (finished teachers are skipped). |
| 3 | `04_timing_and_checks` | GPU T4 x2 | `timing/baseline.json` gives hours per run; `status` shows `bn_micro1/4`, `bn_stock4`, `baseline_s0` done (re-run until then). |
| 4 | `06_evaluate` | None | BatchNorm check PASS; baseline score vs published (step 8 gate). |
| 5 | `05_train_queue` with `QUEUE = "queues/main.yaml"` | GPU T4 x2 | Re-run (same or another account) until all 15 jobs are done. |
| 6 | `05_train_queue` with `queues/ablations.yaml` | GPU T4 x2 | 24 short jobs done. |
| 7 | `02_more_datasets`, then `03` steps for NYUD/MultiCue, then `05` with `queues/nyud.yaml` / `queues/multicue.yaml` | | |
| 8 | `06_evaluate` | None | `results/tables.md`, `agreement_bins.json`, `efficiency.json` on HF. |

**Check progress any time** (from a notebook or a laptop with `huggingface_hub`):
`python -m pedger_kd.sync --remote hf:<HF_REPO> status`, or open `runs/<run>/progress.json` on the HF repo page.
Never start two sessions on the same queue within ~2 minutes of each other (claims are staggered, not atomic).
`--take-over` forces a claim if you are sure the other session is dead.

## Repository map

```
pedger_kd/   upstream.py (loads official PEdger)  norm.py (per-image BN)  augment.py  data.py
             agreement.py  soft_target.py (Variant A)  kd_loss.py (Variant B)  losses.py
             train_kd.py (resumable trainer)  jobs.py (queue runner)  sync.py (HF remote)
             inference.py  evaluation.py  config.py  teachers.py
queues/      checks.yaml  main.yaml  ablations.yaml  nyud.yaml  multicue.yaml
data/        prepare_bsds.py  prepare_list.py  make_multicue_splits.py  visualize_samples.py
teachers/    export_images.py  run_sauge.py  run_muge.py  pack_cache.py  ensemble.py  subset_cache.py  teacher_gate.py
eval/        run_bsds_eval.py  eval_runs.py (score all finished runs)  bn_check.py  agreement_bins.py
bench/       time_run.py  speed_flops.py
results/     make_tables.py  published.yaml
configs/     base.yaml + main rows, ablations/, checks/, nyud/, multicue/  (generated by tools/make_configs.py)
notebooks/   01-06 Kaggle notebooks (generated by tools/make_notebooks.py)
tests/       sanity tests, equivalence tests, augmentation alignment, data prep, end-to-end smoke test
tools/       fetch_upstream.py (pinned commits)  make_configs.py  make_notebooks.py  make_code_zip.py
```

## Local setup

```
python tools/fetch_upstream.py          # official PEdger at the pinned commit
pip install -r requirements.txt
pytest tests                            # 82 tests, CPU, ~5 minutes
```
`pyEdgeEval` builds a C++ extension. It installs out of the box on Kaggle/Linux; on Windows it needs
the MSVC build tools. Everything except benchmark scoring runs without it.
