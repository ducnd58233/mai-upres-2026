import logging
import math
import os
import warnings
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.losses import dct_l1_loss, psnr_255

logger = logging.getLogger(__name__)


@torch.no_grad()
def validate_psnr(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_images: int = 0,
) -> float:
    model.eval()
    psnrs: list[float] = []
    seen = 0
    for lr, hr in loader:
        lr = lr.to(device)
        hr = hr.to(device)
        sr = model(lr)
        sr = torch.clamp(sr, 0.0, 255.0)
        psnrs.append(psnr_255(sr, hr))
        seen += 1
        if max_images > 0 and seen >= max_images:
            break
    return float(np.mean(psnrs)) if psnrs else 0.0


def channel_shuffle_rgb(
    lr: torch.Tensor, hr: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    perm = torch.randperm(3, device=lr.device)
    return lr[:, perm], hr[:, perm]


@torch.no_grad()
def apply_weight_clipping(
    model: torch.nn.Module,
    clip_other: float = 2.0,
    clip_rep: float = 3.0,
) -> None:
    for name, p in model.named_parameters():
        if not name.endswith(".weight") or p.ndim != 4:
            continue
        lim = clip_rep if "rep." in name else clip_other
        p.clamp_(-lim, lim)


def make_cosine_warmup_scheduler(
    opt: torch.optim.Optimizer,
    total_epochs: int,
    warmup_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_epochs = max(1, int(total_epochs * warmup_ratio))

    def lr_lambda(ep: int) -> float:
        if ep < warmup_epochs:
            return float(ep + 1) / float(warmup_epochs)
        t = (ep - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def make_step_halve_scheduler(
    opt: torch.optim.Optimizer,
    step_size: int = 40,
    gamma: float = 0.5,
) -> torch.optim.lr_scheduler.StepLR:
    return torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)


def _qat_backends(backend: str) -> list[str]:
    order = [backend, "x86", "fbgemm", "qnnpack"]
    return list(dict.fromkeys(order))


def _supported_qat_engines() -> list[str] | None:
    try:
        return list(torch.backends.quantized.supported_engines)
    except Exception:
        return None


def prepare_qat(model: torch.nn.Module, backend: str = "qnnpack") -> torch.nn.Module:
    import torch.ao.quantization as tq

    preferred = _qat_backends(backend)
    supported = _supported_qat_engines()
    if supported is not None:
        backends_to_try = [b for b in preferred if b in supported]
    else:
        backends_to_try = preferred

    active_backend: str | None = None
    last_err: RuntimeError | None = None

    for eng in backends_to_try:
        try:
            torch.backends.quantized.engine = eng
            active_backend = eng
            break
        except RuntimeError as e:
            last_err = e
            if supported is not None:
                raise
            if "not supported" in str(e).lower() or "not available" in str(e).lower():
                continue
            raise

    if active_backend is None:
        logger.warning(
            "No QAT backend available (tried: %s).%s Stage 3 will train in fp32. Set qat=False to silence.",
            ", ".join(backends_to_try),
            f" (last error: {last_err})" if last_err else "",
        )
        model.train()
        return model

    model.train()
    setattr(model, "qconfig", tq.get_default_qat_qconfig(active_backend))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        tq.prepare_qat(model, inplace=True)
    return model


def strip_qat_to_clean_state(clean_model: torch.nn.Module, qat_state: dict) -> dict:
    clean_sd = clean_model.state_dict()
    return {
        k: v
        for k, v in qat_state.items()
        if (k in clean_sd and v.shape == clean_sd[k].shape)
    }


def load_ckpt_weights(
    model: torch.nn.Module, ckpt_path: str, device: torch.device
) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    return ckpt


def train_stage(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    base_lr: float,
    out_dir: str,
    start_epoch: int,
    val_max: int,
    stage_tag: str,
    grad_clip: float = 0.0,
    scheduler_type: str = "none",
    channel_shuffle: bool = False,
    loss_mode: str = "l1",
    weight_clip: bool = True,
    wc_other: float = 2.0,
    wc_rep: float = 3.0,
    val_every: int = 1,
) -> str:
    import torch.nn.functional as F

    opt = torch.optim.Adam(model.parameters(), lr=base_lr)

    if scheduler_type == "cos_warmup":
        sch: Optional[torch.optim.lr_scheduler.LRScheduler] = (
            make_cosine_warmup_scheduler(opt, total_epochs=epochs, warmup_ratio=0.1)
        )
    elif scheduler_type == "step_halve":
        sch = make_step_halve_scheduler(opt, step_size=40, gamma=0.5)
    else:
        sch = None

    best = -1e9
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, f"ckpt_best_{stage_tag}.pt")
    last_path = os.path.join(out_dir, f"ckpt_last_{stage_tag}.pt")

    logger.info(
        "[%s] Starting %d epochs (global %d .. %d)",
        stage_tag,
        epochs,
        start_epoch,
        start_epoch + epochs - 1,
    )

    for local_ep in range(epochs):
        ep = start_epoch + local_ep
        current = local_ep + 1
        model.train()
        pbar = tqdm(
            train_loader,
            desc=f"[{stage_tag}] epoch {current}/{epochs} (lr={opt.param_groups[0]['lr']:.2e})",
        )
        losses: list[float] = []

        for lr_img, hr_img in pbar:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            if channel_shuffle:
                lr_img, hr_img = channel_shuffle_rgb(lr_img, hr_img)

            opt.zero_grad(set_to_none=True)
            sr = model(lr_img)

            if loss_mode == "dct":
                loss = dct_l1_loss(sr, hr_img)
            elif loss_mode == "l2":
                loss = F.mse_loss(sr, hr_img)
            else:
                loss = F.l1_loss(sr, hr_img)

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            if weight_clip:
                apply_weight_clipping(model, clip_other=wc_other, clip_rep=wc_rep)

            losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(losses)))

        if sch is not None:
            sch.step()

        do_val = (
            (val_every <= 1)
            or ((local_ep + 1) % val_every == 0)
            or (local_ep == epochs - 1)
        )
        if do_val:
            val_psnr = validate_psnr(model, val_loader, device, max_images=val_max)
            logger.info(
                "[val/%s] epoch %d/%d PSNR=%.4f",
                stage_tag,
                current,
                epochs,
                val_psnr,
            )

            ckpt = {
                "epoch": ep,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "val_psnr": val_psnr,
                "stage": stage_tag,
            }
            torch.save(ckpt, last_path)

            if val_psnr > best:
                best = val_psnr
                torch.save(ckpt, best_path)
                logger.info("  -> saved best (%s)", stage_tag)

    return best_path
