"""Task 6: image, label, agreement and teacher maps stay aligned after augmentation."""
import math

import torch

from pedger_kd.augment import IMAGENET_MEAN, IMAGENET_STD, JointAugment, largest_rotated_rect

NO_COLOR = {"color_jitter": None, "grayscale_p": 0.0}


def synthetic(B=3, H=60, W=90):
    torch.manual_seed(0)
    m = torch.zeros(B, H, W)
    m[:, 15, 5:80] = 1
    m[:, 5:55, 30] = 1
    m[:, 40:50, 60:75] = 1
    m[1] = m[1].roll(7, 1)
    img = (m[..., None].repeat(1, 1, 1, 3) * 255).to(torch.uint8)
    T = (m * 255).to(torch.uint8)
    return img, m, T


def test_all_channels_identical_after_every_transform():
    img, m, T = synthetic()
    aug = JointAugment(NO_COLOR)
    for angle in aug.angles:
        for scale in aug.scales:
            out = aug(img, m, m, T, angle=angle, scale=scale)
            im = out["img"][:, 0:1] * IMAGENET_STD[0] + IMAGENET_MEAN[0]
            assert out["img"].shape[-2:] == out["a"].shape[-2:] == out["T"].shape[-2:]
            assert torch.allclose(im, out["a"], atol=1e-4), (angle, scale)
            assert torch.allclose(out["T"], out["a"], atol=1e-4), (angle, scale)
            assert torch.allclose(out["a_alpha"], out["a"], atol=1e-6)
            assert torch.equal(out["Y"], (out["a"] >= 0.2).float())


def test_rotated_crop_has_no_outside_pixels():
    W, H = 481, 321
    for deg in [22.5, 45, 67.5, 10, 80]:
        th = math.radians(deg)
        wr, hr = largest_rotated_rect(W, H, th)
        c, s = math.cos(th), math.sin(th)
        for sx in (-1, 1):
            for sy in (-1, 1):
                x, y = sx * wr / 2, sy * hr / 2
                xi, yi = c * x + s * y, -s * x + c * y
                assert abs(xi) <= W / 2 + 1e-6 and abs(yi) <= H / 2 + 1e-6, deg


def test_epoch_size_matches_hed_augmentation():
    assert JointAugment({}).num_variants() == 96   # 16 angles x 3 scales x 2 flips
    assert 300 * JointAugment({}).num_variants() == 28800


def test_colour_jitter_keeps_geometry_and_labels():
    img, m, T = synthetic()
    torch.manual_seed(1)
    out = JointAugment({})(img, m, m, T, angle=22.5, scale=1.0)
    torch.manual_seed(1)
    out_nc = JointAugment(NO_COLOR)(img, m, m, T, angle=22.5, scale=1.0)
    assert out["img"].shape == out_nc["img"].shape
    assert torch.equal(out["a"], out_nc["a"]) and torch.equal(out["T"], out_nc["T"])
