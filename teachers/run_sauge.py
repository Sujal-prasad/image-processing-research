"""Run SAUGE (AAAI 2025, frozen SAM ViT-B + 1.3M side network) on a folder of images.

Saves the "definite" output (SAUGE's main edge map) as 16-bit PNGs.
Weights: third_party/SAUGE/ckpt/bsds/sauge_vitb.pth (in the repo) and SAM ViT-B:
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
Requires: einops (plus the SAM dependencies, already vendored inside SAUGE).

    python teachers/run_sauge.py --images work/img_trainval --out work/pred_sauge \
        --sam-ckpt sam_vit_b_01ec64.pth
"""
import argparse
import sys
import types
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import crop_for_model, list_images, read_rgb01, save_prob16, uncrop  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sauge-dir", default=str(ROOT / "third_party" / "SAUGE"))
    p.add_argument("--ckpt", default=None, help="default: <sauge-dir>/ckpt/bsds/sauge_vitb.pth")
    p.add_argument("--sam-ckpt", required=True)
    p.add_argument("--fit", default="bsds_crop", choices=["bsds_crop", "pad32", "none"])
    p.add_argument("--save-granularities", action="store_true",
                   help="also save SAUGE's coarse/middle/fine outputs")
    args = p.parse_args()
    sauge_dir = Path(args.sauge_dir)
    sys.path.insert(0, str(sauge_dir))
    from segment_anything import sam_model_registry, SamAutomaticMaskGeneratorForTest  # noqa: E402
    import importlib
    Model = importlib.import_module("model.sauge_vitb")

    torch.manual_seed(6666)
    sam = sam_model_registry["vit_b"](checkpoint=args.sam_ckpt)
    for prm in sam.parameters():
        prm.requires_grad = False
    sam.cuda()
    gen = SamAutomaticMaskGeneratorForTest(sam)
    model = Model.SAUGE(types.SimpleNamespace(), sam_generator=gen, mode="eval").cuda()
    ckpt = args.ckpt or str(sauge_dir / "ckpt" / "bsds" / "sauge_vitb.pth")
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
    model.eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in tqdm(list_images(args.images)):
        x = read_rgb01(path)
        xc, info = crop_for_model(x, args.fit)
        with torch.no_grad():
            multi, final, _, _ = model(xc.cuda())
        save_prob16(out / f"{path.stem}.png", uncrop(final.float().cpu(), info)[0, 0].numpy())
        if args.save_granularities:
            for gname, g in zip(("coarse", "middle", "fine"), multi):
                gd = out.parent / f"{out.name}_{gname}"
                gd.mkdir(exist_ok=True)
                save_prob16(gd / f"{path.stem}.png", uncrop(g.float().cpu(), info)[0, 0].numpy())


if __name__ == "__main__":
    main()
