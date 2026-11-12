"""Joint GPU augmentation (coding task 6).

Every geometric transform is applied to ONE stacked tensor holding the image, the
annotator-fraction map, the agreement map and the teacher map, so they can never
drift apart. Geometry follows the HED/RCF BSDS augmentation family that PEdger's
training data was built from: 16 rotations (multiples of 22.5 deg), horizontal
flips, scales {0.5, 1, 1.5}. Rotations that are not multiples of 90 deg are
cropped to the largest axis-aligned rectangle inside the rotated image, so no
artificial black borders (= fake edges) appear.

One rotation angle and one scale are shared by a micro-batch (so all images in it
have the same size); flips, colour jitter and grayscale are drawn per image.
After geometry: label Y = (a >= 0.2), exactly PEdger's binarisation.
"""
import math

import torch
import torch.nn.functional as F
import torchvision.transforms as TV

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def largest_rotated_rect(w, h, angle_rad):
    """Size (w, h) of the largest axis-aligned rectangle inside a w x h rectangle rotated by angle."""
    if w <= 0 or h <= 0:
        return 0, 0
    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)
    sin_a, cos_a = abs(math.sin(angle_rad)), abs(math.cos(angle_rad))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a
    return wr, hr


def rotate_crop(x, deg):
    """Rotate (B,C,H,W) by `deg` about the centre and return the largest valid crop."""
    if deg % 360 == 0:
        return x
    B, C, H, W = x.shape
    th = math.radians(deg)
    wr, hr = largest_rotated_rect(W, H, th)
    wr, hr = max(1, int(math.floor(wr))), max(1, int(math.floor(hr)))
    dev = x.device
    ys = torch.arange(hr, device=dev, dtype=torch.float32) - (hr - 1) / 2
    xs = torch.arange(wr, device=dev, dtype=torch.float32) - (wr - 1) / 2
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    c, s = math.cos(th), math.sin(th)
    x_in = c * gx + s * gy
    y_in = -s * gx + c * gy
    grid = torch.stack([2 * x_in / W, 2 * y_in / H], -1)[None].expand(B, -1, -1, -1)
    return F.grid_sample(x, grid.to(x.dtype), mode="bilinear", padding_mode="border", align_corners=False)


class JointAugment:
    def __init__(self, cfg, train=True):
        self.cfg = cfg
        self.train = train
        angles = cfg.get("angles_deg", [i * 22.5 for i in range(16)])
        self.angles = [float(a) for a in angles] if train else [0.0]
        self.scales = [float(s) for s in cfg.get("scales", [0.5, 1.0, 1.5])] if train else [1.0]
        self.hflip = cfg.get("hflip", True) and train
        self.crop = cfg.get("crop")  # None or [h, w] (e.g. MultiCue)
        self.label_threshold = cfg.get("label_threshold", 0.2)
        cj = cfg.get("color_jitter", [0.5, 0.5, 0.5, 0.5])
        ops = []
        if train and cj:
            ops.append(TV.ColorJitter(brightness=cj[0], contrast=cj[1], saturation=cj[2], hue=cj[3]))
        if train and cfg.get("grayscale_p", 0.2) > 0:
            ops.append(TV.RandomGrayscale(p=cfg.get("grayscale_p", 0.2)))
        self.color = TV.Compose(ops) if ops else None

    def num_variants(self):
        """Augmented copies per image in HED-style offline augmentation (defines an epoch)."""
        return len(self.angles) * len(self.scales) * (2 if self.hflip else 1)

    def sample_geometry(self):
        i = int(torch.randint(len(self.angles), (1,)).item())
        j = int(torch.randint(len(self.scales), (1,)).item())
        return self.angles[i], self.scales[j]

    def __call__(self, img_u8, frac, frac_alpha, teacher_u8=None, angle=None, scale=None):
        """img_u8 (B,H,W,3) uint8; frac / frac_alpha (B,H,W) float/half in [0,1];
        teacher_u8 (B,H,W) uint8 or None. All on the same device.
        Returns dict with img (B,3,h,w) normalised, Y, a, a_alpha, T (B,1,h,w) float32."""
        if angle is None or scale is None:
            a0, s0 = self.sample_geometry()
            angle = a0 if angle is None else angle
            scale = s0 if scale is None else scale
        chans = [img_u8.permute(0, 3, 1, 2).float() / 255.0,
                 frac[:, None].float(), frac_alpha[:, None].float()]
        if teacher_u8 is not None:
            chans.append(teacher_u8[:, None].float() / 255.0)
        x = torch.cat(chans, 1)

        if self.hflip:
            flip = torch.rand(x.shape[0], device=x.device) < 0.5
            if flip.any():
                x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        k90, rem = divmod(angle % 360.0, 90.0)
        if k90:
            x = torch.rot90(x, int(k90), dims=(2, 3))
        if rem:
            x = rotate_crop(x, rem)
        if scale != 1.0:
            h, w = x.shape[-2:]
            x = F.interpolate(x, size=(max(1, round(h * scale)), max(1, round(w * scale))),
                              mode="bilinear", align_corners=False)
        if self.crop:
            x = self._random_crop(x, self.crop)

        img = x[:, 0:3].clamp(0, 1)
        a = x[:, 3:4].clamp(0, 1)
        a_alpha = x[:, 4:5].clamp(0, 1)
        T = x[:, 5:6].clamp(0, 1) if teacher_u8 is not None else None
        if self.color is not None:
            img = torch.stack([self.color(im) for im in img], 0)
        img = TV.functional.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
        Y = (a >= self.label_threshold).float()
        return {"img": img.contiguous(), "Y": Y, "a": a, "a_alpha": a_alpha, "T": T,
                "angle": angle, "scale": scale}

    @staticmethod
    def _random_crop(x, crop):
        ch, cw = int(crop[0]), int(crop[1])
        B, C, H, W = x.shape
        ch, cw = min(ch, H), min(cw, W)
        out = []
        for b in range(B):
            y0 = int(torch.randint(H - ch + 1, (1,)).item())
            x0 = int(torch.randint(W - cw + 1, (1,)).item())
            out.append(x[b, :, y0:y0 + ch, x0:x0 + cw])
        return torch.stack(out, 0)


def normalize_image_u8(img_u8_hwc, device):
    """Test-time preprocessing identical to PEdger's test.py (ToTensor + ImageNet normalise)."""
    x = torch.as_tensor(img_u8_hwc, device=device).permute(2, 0, 1).float()[None] / 255.0
    return TV.functional.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
