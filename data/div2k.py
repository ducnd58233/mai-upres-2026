import glob
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from data.constants import (
    DIV2K_TRAIN_HR_PATTERNS,
    DIV2K_TRAIN_LR_PATTERNS_TEMPLATE,
    DIV2K_VALID_HR_PATTERNS,
    DIV2K_VALID_LR_PATTERNS_TEMPLATE,
)


def _pil_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _img_to_tensor_255(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _augment(lr: torch.Tensor, hr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        lr = torch.flip(lr, dims=[2])
        hr = torch.flip(hr, dims=[2])
    if random.random() < 0.5:
        lr = torch.flip(lr, dims=[1])
        hr = torch.flip(hr, dims=[1])
    if random.random() < 0.5:
        lr = lr.transpose(1, 2)
        hr = hr.transpose(1, 2)
    return lr, hr


class DIV2KPairX3(Dataset):
    """
    Paired DIV2K: HR {id}.png, LR {id}x{scale}.png.
    Train: random crop LR patch; val: full image.
    """

    def __init__(
        self,
        hr_dir: str,
        lr_x3_dir: str,
        train: bool = True,
        scale: int = 3,
        lr_patch: int = 128,
        augment: bool = True,
        repeat: int = 1,
    ):
        super().__init__()
        self.hr_dir = hr_dir
        self.lr_dir = lr_x3_dir
        self.train = train
        self.scale = scale
        self.lr_patch = lr_patch
        self.augment = augment and train
        self.repeat = max(1, int(repeat))

        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
        if not self.hr_paths:
            raise FileNotFoundError(f"No HR png found in: {hr_dir}")

        for p in self.hr_paths[:10]:
            stem = os.path.splitext(os.path.basename(p))[0]
            lr_p = os.path.join(self.lr_dir, f"{stem}x{scale}.png")
            if not os.path.exists(lr_p):
                raise FileNotFoundError(f"Missing LR pair for {p} -> expected {lr_p}")

    def __len__(self) -> int:
        return len(self.hr_paths) * self.repeat

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr_path = self.hr_paths[idx % len(self.hr_paths)]
        stem = os.path.splitext(os.path.basename(hr_path))[0]
        lr_path = os.path.join(self.lr_dir, f"{stem}x{self.scale}.png")

        hr = _img_to_tensor_255(_pil_rgb(hr_path))
        lr = _img_to_tensor_255(_pil_rgb(lr_path))

        _, h_lr, w_lr = lr.shape
        exp_h_hr, exp_w_hr = h_lr * self.scale, w_lr * self.scale
        hr = hr[:, :exp_h_hr, :exp_w_hr]

        if not self.train:
            return lr, hr

        ps = self.lr_patch
        if h_lr < ps or w_lr < ps:
            pad_h = max(0, ps - h_lr)
            pad_w = max(0, ps - w_lr)
            lr = F.pad(lr, (0, pad_w, 0, pad_h), mode="reflect")
            hr = F.pad(
                hr, (0, pad_w * self.scale, 0, pad_h * self.scale), mode="reflect"
            )
            _, h_lr, w_lr = lr.shape

        x = random.randint(0, w_lr - ps)
        y = random.randint(0, h_lr - ps)

        lr_crop = lr[:, y : y + ps, x : x + ps]
        hr_crop = hr[
            :,
            y * self.scale : (y + ps) * self.scale,
            x * self.scale : (x + ps) * self.scale,
        ]

        if self.augment:
            lr_crop, hr_crop = _augment(lr_crop, hr_crop)

        return lr_crop, hr_crop


def _find_dir_candidates(root: str, patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        cand = os.path.join(root, pat)
        if os.path.isdir(cand):
            return cand
    for pat in patterns:
        if "*" in pat:
            hits = glob.glob(os.path.join(root, pat))
            hits = [h for h in hits if os.path.isdir(h)]
            if hits:
                return hits[0]
    return None


def resolve_div2k_paths(data_root: str, scale: int = 3) -> Tuple[str, str, str, str]:
    """Resolve DIV2K train_HR, train_LR, valid_HR, valid_LR under data_root."""
    train_lr_pat = [p.format(scale=scale) for p in DIV2K_TRAIN_LR_PATTERNS_TEMPLATE]
    valid_lr_pat = [p.format(scale=scale) for p in DIV2K_VALID_LR_PATTERNS_TEMPLATE]

    train_hr = _find_dir_candidates(data_root, DIV2K_TRAIN_HR_PATTERNS)
    valid_hr = _find_dir_candidates(data_root, DIV2K_VALID_HR_PATTERNS)
    train_lr = _find_dir_candidates(data_root, train_lr_pat)
    valid_lr = _find_dir_candidates(data_root, valid_lr_pat)

    if not train_hr or not valid_hr or not train_lr or not valid_lr:
        raise FileNotFoundError(
            "Cannot auto-resolve DIV2K paths. Found:\n"
            f"  train_hr={train_hr}\n  train_lr={train_lr}\n  valid_hr={valid_hr}\n  valid_lr={valid_lr}\n"
            "Please check your folder names and update patterns in resolve_div2k_paths()."
        )

    return train_hr, train_lr, valid_hr, valid_lr
