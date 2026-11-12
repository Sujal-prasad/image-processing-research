"""Generate the experiment configs (everything except configs/base.yaml).

Run once after editing the definitions below:  python tools/make_configs.py
"""
from pathlib import Path

import yaml

C = Path(__file__).resolve().parents[1] / "configs"

FILES = {
    # ---------------------------------------------------------------- main rows (3 seeds)
    "baseline.yaml": ("Official PEdger retrained in our pipeline (step 8). No teachers.",
                      {"base": "base.yaml", "name": "baseline"}),
    "B_uniform.yaml": ("Variant B, uniform weight, SAUGE only (H1).",
                       {"base": "base.yaml", "name": "B_uniform", "data": {"teacher": "sauge"},
                        "kd": {"enabled": True, "weight": "uniform"}}),
    "B_agree.yaml": ("Variant B, agreement-weighted, teacher ensemble.",
                     {"base": "base.yaml", "name": "B_agree", "data": {"teacher": "ensemble_mean"},
                      "kd": {"enabled": True, "weight": "agreement"}}),
    "A_uniform.yaml": ("Variant A with uniform alpha = 1 - eta (PEdger's schedule, same for every pixel) and the\n"
                       "teacher ensemble mixed in. Control for H2: same as A_agree except that trust does not\n"
                       "depend on human agreement.",
                       {"base": "base.yaml", "name": "A_uniform", "data": {"teacher": "ensemble_mean"},
                        "method": {"target": "agreement", "alpha_mode": "eta", "ramp": False}}),
    "A_agree.yaml": ("Variant A, agreement-aware, teacher ensemble: the full method.",
                     {"base": "base.yaml", "name": "A_agree", "data": {"teacher": "ensemble_mean"},
                      "method": {"target": "agreement", "alpha_mode": "agreement", "alpha_min": 0.1,
                                 "gamma": 0.5, "ramp": True}}),
    # ---------------------------------------------------------------- short ablation schedule (1 seed)
    "ablations/_short.yaml": ("Short ablation schedule: dev split (train, 200 images; score on val), one third of the\n"
                              "samples per epoch. Epoch count, warm-up, eta ramp and momentum averaging are unchanged.\n"
                              "Check once that short runs rank methods like full runs (short_check_* vs full_check_*).",
                              {"base": "../A_agree.yaml", "data": {"split": "train", "samples_per_epoch": 6400},
                               "train": {"schedule_tag": "short"}}),
    "ablations/_short_B.yaml": ("Short schedule for Variant B ablations.",
                                {"base": "../B_uniform.yaml", "data": {"split": "train", "samples_per_epoch": 6400},
                                 "train": {"schedule_tag": "short"}}),
    # ---------------------------------------------------------------- BatchNorm check
    "checks/_bn.yaml": ("BatchNorm check (step 9): short baseline runs on the dev split, scored on val.",
                        {"base": "../baseline.yaml", "data": {"split": "train", "samples_per_epoch": 3200},
                         "train": {"schedule_tag": "bn_check"}}),
    "checks/bn_micro1.yaml": ("PEdger's setup: 1 image per forward pass, 16 accumulated.",
                              {"base": "_bn.yaml", "name": "bn_micro1", "train": {"micro_batch": 1, "accum_steps": 16}}),
    "checks/bn_micro4.yaml": ("Our setup: 4 images per pass with per-image normalisation.",
                              {"base": "_bn.yaml", "name": "bn_micro4",
                               "train": {"micro_batch": 4, "accum_steps": 4, "norm": "per_sample"}}),
    "checks/bn_stock4.yaml": ("What to avoid: 4 images per pass with stock BN (statistics pooled across images).",
                              {"base": "_bn.yaml", "name": "bn_stock4",
                               "train": {"micro_batch": 4, "accum_steps": 4, "norm": "stock"}}),
    # ---------------------------------------------------------------- NYUDv2
    "nyud/_nyud.yaml": ("NYUDv2: one annotation per image, so no agreement map. Store from data/prepare_list.py.",
                        {"base": "../base.yaml", "data": {"store": "nyud", "teacher_dir": "nyud", "split": "trainval"}}),
    "nyud/baseline.yaml": ("", {"base": "_nyud.yaml", "name": "nyud_baseline"}),
    "nyud/A_eta.yaml": ("(a) constant alpha following PEdger's eta schedule, teachers mixed in.",
                        {"base": "_nyud.yaml", "name": "nyud_A_eta", "data": {"teacher": "ensemble_mean"},
                         "method": {"target": "agreement", "alpha_mode": "eta", "ramp": False}}),
    "nyud/A_teacher_conf.yaml": ("(b) teacher-confidence trust: alpha = 1 - c |2T - 1|.",
                                 {"base": "_nyud.yaml", "name": "nyud_A_teacher_conf",
                                  "data": {"teacher": "ensemble_mean"},
                                  "method": {"target": "agreement", "alpha_mode": "teacher_conf",
                                             "teacher_conf_c": 0.8, "ramp": True}}),
    "nyud/B_uniform.yaml": ("", {"base": "_nyud.yaml", "name": "nyud_B_uniform", "data": {"teacher": "ensemble_mean"},
                                 "kd": {"enabled": True, "weight": "uniform"}}),
    # ---------------------------------------------------------------- MultiCue boundary
    "multicue/_multicue.yaml": ("MultiCue boundaries, split 0 of 3 (override data.store / data.teacher_dir for\n"
                                "splits 1 and 2). Images are 1280x720: random 512x512 crops.",
                                {"base": "../base.yaml",
                                 "data": {"store": "multicue_boundary_split0", "teacher_dir": "multicue_boundary_split0",
                                          "split": "train"},
                                 "augment": {"crop": [512, 512]}}),
    "multicue/baseline.yaml": ("", {"base": "_multicue.yaml", "name": "multicue_baseline"}),
    "multicue/A_agree.yaml": ("", {"base": "_multicue.yaml", "name": "multicue_A_agree",
                                   "data": {"teacher": "ensemble_mean"},
                                   "method": {"target": "agreement", "alpha_mode": "agreement", "alpha_min": 0.1,
                                              "gamma": 0.5, "ramp": True}}),
}

ABL_A = {"teacher_sauge": {"data": {"teacher": "sauge"}}, "teacher_muge": {"data": {"teacher": "muge"}},
         "teacher_memo": {"data": {"teacher": "memo"}}, "teacher_ensemble_mean": {},
         "teacher_ensemble_conf": {"data": {"teacher": "ensemble_confidence"}},
         "gamma_0": {"method": {"gamma": 0.0}}, "gamma_1": {"method": {"gamma": 1.0}},
         "gamma_conf": {"method": {"gamma": "confidence"}}, "no_ramp": {"method": {"ramp": False}},
         "alpha_min_0": {"method": {"alpha_min": 0.0}}, "alpha_min_0.3": {"method": {"alpha_min": 0.3}},
         "dilation_1": {"data": {"agreement_dilation": 1}}, "dilation_2": {"data": {"agreement_dilation": 2}}}
ABL_B = {"B_lambda_0.5": {"kd": {"lambda": 0.5}}, "B_lambda_2": {"kd": {"lambda": 2.0}},
         "B_tau_1": {"kd": {"tau": 1.0}}, "B_tau_4": {"kd": {"tau": 4.0}}, "B_no_ls": {"kd": {"use_ls": False}}}
for k, v in ABL_A.items():
    FILES[f"ablations/{k}.yaml"] = ("", {"base": "_short.yaml", "name": f"abl_{k}", **v})
for k, v in ABL_B.items():
    FILES[f"ablations/{k}.yaml"] = ("", {"base": "_short_B.yaml", "name": f"abl_{k}", **v})
for m in ("baseline", "B_uniform", "A_agree"):
    FILES[f"ablations/short_check_{m}.yaml"] = (
        f"Short-schedule {m} on the dev split; pairs with full_check_{m}.",
        {"base": f"../{m}.yaml", "name": f"short_check_{m}", "data": {"split": "train", "samples_per_epoch": 6400},
         "train": {"schedule_tag": "short"}})
    FILES[f"ablations/full_check_{m}.yaml"] = (
        f"Full-schedule {m} on the dev split; pairs with short_check_{m}.",
        {"base": f"../{m}.yaml", "name": f"full_check_{m}", "data": {"split": "train"},
         "train": {"schedule_tag": "full_dev"}})


def main():
    for rel, (comment, body) in FILES.items():
        path = C / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        head = "".join(f"# {line}\n" for line in comment.splitlines()) if comment else ""
        path.write_text(head + yaml.safe_dump(body, sort_keys=False))
    print(f"wrote {len(FILES)} configs under {C}")


if __name__ == "__main__":
    main()
