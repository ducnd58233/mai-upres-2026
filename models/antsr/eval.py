from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.div2k import DIV2KPairX3, resolve_div2k_paths
from models.antsr.model import AntSR
from training.stage import validate_psnr

logger = logging.getLogger(__name__)


def run_antsr_eval_pytorch(
    checkpoint_path: str,
    data_root: str,
    device: torch.device,
    max_images: int = 0,
    batch_size: int = 1,
    workers: int = 0,
) -> float:
    """Load AntSR checkpoint, run on DIV2K valid, return mean PSNR (dB)."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt or "cfg" not in ckpt:
        raise ValueError(
            "Expected AntSR checkpoint dict with 'model' and 'cfg'. "
            f"Got keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
        )
    cfg = dict(ckpt["cfg"])
    state = ckpt["model"]

    model = AntSR(deploy=True, **cfg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    train_hr, train_lr, valid_hr, valid_lr = resolve_div2k_paths(data_root, scale=3)
    val_ds = DIV2KPairX3(
        valid_hr, valid_lr, train=False, lr_patch=128, augment=False, repeat=1
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )

    psnr_val = validate_psnr(model, val_loader, device, max_images=max_images)
    logger.info("AntSR PyTorch eval PSNR: %.4f dB", psnr_val)
    return psnr_val
