"""Run MuGE (CVPR 2024, EfficientNet-B7 UNet++, granularity-controllable) on a folder of images.

MuGE predicts a Gaussian over logits for a requested granularity alpha. The
official test script samples from it; for a deterministic soft teacher map we use
sigmoid(mean) and average over the granularities (plan 8.4: "the average of its
granularity outputs").
Weights ("BSDS with scale for MuGE", Google Drive id 15NucsEeHAFwo5O2s11pMR1BikoUtiuUX):
    gdown 15NucsEeHAFwo5O2s11pMR1BikoUtiuUX -O muge_bsds.pth
Requires: efficientnet_pytorch.

    python teachers/run_muge.py --images work/img_trainval --out work/pred_muge --ckpt muge_bsds.pth
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
    p.add_argument("--uaed-dir", default=str(ROOT / "third_party" / "UAED"))
    p.add_argument("--ckpt", required=True)
    p.add_argument("--alphas", type=float, nargs="+", default=[i / 10 for i in range(11)])
    p.add_argument("--fit", default="bsds_crop", choices=["bsds_crop", "pad32", "none"])
    args = p.parse_args()
    sys.path.insert(0, str(Path(args.uaed_dir)))
    from model.sigma_logit_unetpp_alpha_ffthalf_feat import Mymodel  # noqa: E402

    torch.manual_seed(0)
    model = Mymodel(types.SimpleNamespace(distribution="gs"), encoder_weights=None).cuda()
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd.get("state_dict", sd))
    model.eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in tqdm(list_images(args.images)):
        x = read_rgb01(path)
        xc, info = crop_for_model(x, args.fit)
        xc = xc.cuda()
        acc = 0
        with torch.no_grad():
            for a in args.alphas:
                mean, _std = model(xc, torch.ones(1, device="cuda") * a)
                acc = acc + torch.sigmoid(mean.float())
        prob = acc / len(args.alphas)
        save_prob16(out / f"{path.stem}.png", uncrop(prob.cpu(), info)[0, 0].numpy())


if __name__ == "__main__":
    main()
