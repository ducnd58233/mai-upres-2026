import os, glob, random
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def _list_images(root: str):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(root, "**", e), recursive=True)
    return sorted(paths)


def pil_rgb(p: str) -> Image.Image:
    return Image.open(p).convert("RGB")


def to_chw01(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def augment(lr: torch.Tensor, hr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        lr = torch.flip(lr, dims=[2]); hr = torch.flip(hr, dims=[2])
    if random.random() < 0.5:
        lr = torch.flip(lr, dims=[1]); hr = torch.flip(hr, dims=[1])
    if random.random() < 0.5:
        lr = lr.transpose(1, 2); hr = hr.transpose(1, 2)
    return lr, hr


def _interp_bicubic(x, size):
    # antialias exists in newer torch; keep compatibility
    try:
        return F.interpolate(x, size=size, mode="bicubic", align_corners=False, antialias=True)
    except TypeError:
        return F.interpolate(x, size=size, mode="bicubic", align_corners=False)


def bicubic_downsample(hr01: torch.Tensor, scale: int) -> torch.Tensor:
    x = hr01.unsqueeze(0)
    H, W = x.shape[-2:]
    h_lr, w_lr = H // scale, W // scale
    x = x[..., :h_lr * scale, :w_lr * scale]
    lr = _interp_bicubic(x, size=(h_lr, w_lr))
    return lr.squeeze(0)


class RandomHRMixOnTheFlyLR(Dataset):
    def __init__(
        self,
        hr_dirs: List[str],
        scale: int = 3,
        lr_patch: int = 64,
        augment_on: bool = True,
        num_samples: int = 2000000,
        weights: List[float] = None,
    ):
        self.scale = scale
        self.lr_patch = lr_patch
        self.augment_on = augment_on
        self.num_samples = int(num_samples)

        self.groups = []
        for d in hr_dirs:
            paths = _list_images(d)
            if len(paths) == 0:
                raise FileNotFoundError(f"No images found in: {d}")
            self.groups.append(paths)

        if weights is None:
            weights = [1.0 / len(self.groups)] * len(self.groups)
        s = sum(weights)
        self.weights = [w / s for w in weights]

    def __len__(self):
        return self.num_samples

    def _sample_path(self) -> str:
        gi = random.choices(range(len(self.groups)), weights=self.weights, k=1)[0]
        return random.choice(self.groups[gi])

    def __getitem__(self, idx: int):
        p = self._sample_path()
        hr = to_chw01(pil_rgb(p))  # CHW 0..1

        _, H, W = hr.shape
        H2 = (H // self.scale) * self.scale
        W2 = (W // self.scale) * self.scale
        hr = hr[:, :H2, :W2]
        _, H, W = hr.shape

        ps = self.lr_patch
        hs = ps * self.scale
        if H < hs or W < hs:
            pad_h = max(0, hs - H)
            pad_w = max(0, hs - W)
            hr = F.pad(hr, (0, pad_w, 0, pad_h), mode="reflect")
            _, H, W = hr.shape

        x = random.randint(0, W - hs)
        y = random.randint(0, H - hs)
        hr_crop = hr[:, y:y + hs, x:x + hs]
        lr_crop = bicubic_downsample(hr_crop, self.scale)

        if self.augment_on:
            lr_crop, hr_crop = augment(lr_crop, hr_crop)

        return lr_crop, hr_crop