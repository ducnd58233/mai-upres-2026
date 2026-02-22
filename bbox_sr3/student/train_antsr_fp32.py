# train_antsr_fp32.py
# Full training pipeline (FP32 -> fine-tune -> optional QAT) for x3 SR on DIV2K
# Includes "full validation" per epoch:
#   PSNR on RGB + Y, for shave=0 and shave=3 (configurable via --val_shaves)
# Optional: SSIM (RGB) for shave=0/3 (enable via --report_ssim)

import os
import math
import argparse
import random
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Tuple, List, Any
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_antsr import AntSR
from data_div2k_pairs import DIV2KPairX3, resolve_div2k_paths


# -------------------------
# Seed
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Metrics helpers
# -------------------------
def _rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    # x: NCHW, 0..255
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _shave_border(a: torch.Tensor, shave: int) -> torch.Tensor:
    if shave <= 0:
        return a
    h, w = a.shape[-2], a.shape[-1]
    if h <= 2 * shave or w <= 2 * shave:
        return a
    return a[..., shave:-shave, shave:-shave]


def psnr_255(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-12) -> float:
    mse = torch.mean((pred - gt) ** 2).item()
    if mse < eps:
        return 99.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)


# -------------------------
# Optional SSIM (simple, stable)
# - This is not "MS-SSIM", just SSIM. Use only for debugging.
# -------------------------
def _gaussian_1d(win: int, sigma: float, device, dtype):
    coords = torch.arange(win, device=device, dtype=dtype) - (win - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    return g


def _ssim_per_channel(img1: torch.Tensor, img2: torch.Tensor, win: int = 11, sigma: float = 1.5) -> torch.Tensor:
    # img1,img2: NCHW in [0..255]
    # returns: N,C (mean SSIM over H,W)
    device, dtype = img1.device, img1.dtype
    g1 = _gaussian_1d(win, sigma, device, dtype).view(1, 1, 1, win)
    g2 = _gaussian_1d(win, sigma, device, dtype).view(1, 1, win, 1)

    def blur(x):
        C = x.size(1)
        x = F.conv2d(x, g1.expand(C, 1, 1, win), padding=(0, win // 2), groups=C)
        x = F.conv2d(x, g2.expand(C, 1, win, 1), padding=(win // 2, 0), groups=C)
        return x

    mu1 = blur(img1)
    mu2 = blur(img2)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = blur(img1 * img1) - mu1_sq
    sigma2_sq = blur(img2 * img2) - mu2_sq
    sigma12   = blur(img1 * img2) - mu12

    L = 255.0
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    num = (2 * mu12 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / (den + 1e-12)
    return ssim_map.mean(dim=(-1, -2))


@torch.no_grad()
def validate_metrics_all(
    model,
    loader,
    device,
    scale: int,
    max_images: int = 0,
    shaves: Tuple[int, ...] = (0, 3),
    report_ssim: bool = False,
    ssim_win: int = 11,
    ssim_sigma: float = 1.5,
) -> Dict[str, float]:
    model.eval()
    shaves = tuple(sorted(set(int(s) for s in shaves)))

    acc: Dict[str, List[float]] = {}
    def _push(k: str, v: float):
        acc.setdefault(k, []).append(v)

    seen = 0
    for lr, hr in loader:
        lr = lr.to(device)
        hr = hr.to(device)

        bi = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
        sr = model(lr)

        H, W = hr.shape[-2], hr.shape[-1]
        sr = sr[..., :H, :W]
        bi = bi[..., :H, :W]

        for sh in shaves:
            sr_s = _shave_border(sr, sh)
            bi_s = _shave_border(bi, sh)
            hr_s = _shave_border(hr, sh)

            sr_rgb = torch.clamp(sr_s, 0.0, 255.0)
            bi_rgb = torch.clamp(bi_s, 0.0, 255.0)
            hr_rgb = torch.clamp(hr_s, 0.0, 255.0)

            _push(f"psnr_sr_rgb_sh{sh}", psnr_255(sr_rgb, hr_rgb))
            _push(f"psnr_bi_rgb_sh{sh}", psnr_255(bi_rgb, hr_rgb))

            sr_y = torch.clamp(_rgb_to_y(sr_s), 0.0, 255.0)
            bi_y = torch.clamp(_rgb_to_y(bi_s), 0.0, 255.0)
            hr_y = torch.clamp(_rgb_to_y(hr_s), 0.0, 255.0)

            _push(f"psnr_sr_y_sh{sh}", psnr_255(sr_y, hr_y))
            _push(f"psnr_bi_y_sh{sh}", psnr_255(bi_y, hr_y))

            if report_ssim:
                ssim_sr = _ssim_per_channel(sr_rgb, hr_rgb, win=ssim_win, sigma=ssim_sigma).mean().item()
                ssim_bi = _ssim_per_channel(bi_rgb, hr_rgb, win=ssim_win, sigma=ssim_sigma).mean().item()
                _push(f"ssim_sr_rgb_sh{sh}", float(ssim_sr))
                _push(f"ssim_bi_rgb_sh{sh}", float(ssim_bi))

        seen += 1
        if max_images > 0 and seen >= max_images:
            break

    return {k: float(np.mean(v)) if len(v) else 0.0 for k, v in acc.items()}


# -------------------------
# EMA (resume-able)
# -------------------------
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        self._init_from_model(model)

    def _init_from_model(self, model: torch.nn.Module):
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n not in self.shadow:
                self.shadow[n] = p.detach().clone()
                continue
            self.shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

    def apply(self, model: torch.nn.Module):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n].data)

    def restore(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n].data)
        self.backup = {}

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, sd: dict, model: torch.nn.Module):
        self.decay = float(sd.get("decay", self.decay))
        shadow = sd.get("shadow", {})
        self.shadow = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n in shadow:
                self.shadow[n] = shadow[n].to(p.device, dtype=p.dtype).clone()
            else:
                self.shadow[n] = p.detach().clone()


# -------------------------
# DCT loss (FFT-based DCT-II) with CACHE
# -------------------------
_DCT_CACHE = {}

def _get_dct_cache(N: int, device: torch.device, dtype: torch.dtype):
    key = (N, device.type, device.index if device.type == "cuda" else -1, dtype)
    if key in _DCT_CACHE:
        return _DCT_CACHE[key]

    even_idx = torch.arange(0, N, 2, device=device)
    odd_idx  = torch.arange(N - 1, -1, -2, device=device)

    cplx_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    k = torch.arange(N, device=device, dtype=dtype)
    W = torch.exp(-1j * math.pi * k / (2.0 * N)).to(cplx_dtype)

    _DCT_CACHE[key] = {"even": even_idx, "odd": odd_idx, "W": W}
    return _DCT_CACHE[key]

def _dct_1d(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    N = x.size(dim)
    cache = _get_dct_cache(N, x.device, x.dtype)
    even = x.index_select(dim, cache["even"])
    odd  = x.index_select(dim, cache["odd"])
    v = torch.cat([even, odd], dim=dim)
    V = torch.fft.fft(v, dim=dim)
    out = (V * cache["W"]).real * 2.0
    return out

def dct_2d(x: torch.Tensor) -> torch.Tensor:
    x = _dct_1d(x, dim=-1)
    x = _dct_1d(x, dim=-2)
    return x

def dct_l1_loss(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(dct_2d(sr) - dct_2d(hr)))


# -------------------------
# Teacher Cache (KD from precomputed SR)
# -------------------------
class TeacherCache:
    """
    Supports:
      - cache_dir/{stem}.png  (uint8 RGB)
      - cache_dir/{stem}.npy  (HWC float16/float32)
      - cache_dir/{stem}.npz  (arr=HWC float16/float32)
    Returns crops in BCHW float32, range [0..255]
    """
    def __init__(self, cache_dir: str, max_keep: int = 64, prefer: str = "png"):
        self.cache_dir = cache_dir
        self.max_keep = int(max_keep)
        self.prefer = prefer
        self.mem = OrderedDict()  # stem -> np array HWC (float32 or uint8)

    def _load_any(self, stem: str) -> np.ndarray:
        # LRU cache
        if stem in self.mem:
            arr = self.mem.pop(stem)
            self.mem[stem] = arr
            return arr

        p_png = os.path.join(self.cache_dir, f"{stem}.png")
        p_npy = os.path.join(self.cache_dir, f"{stem}.npy")
        p_npz = os.path.join(self.cache_dir, f"{stem}.npz")

        arr = None

        # Prefer png (best compression)
        if self.prefer == "png" and os.path.exists(p_png):
            im = Image.open(p_png).convert("RGB")
            arr = np.array(im)  # uint8 HWC

        elif os.path.exists(p_npy):
            arr = np.load(p_npy, mmap_mode="r")  # HWC float16/32

        elif os.path.exists(p_npz):
            z = np.load(p_npz)
            arr = z["arr"]

        elif os.path.exists(p_png):
            im = Image.open(p_png).convert("RGB")
            arr = np.array(im)

        else:
            raise FileNotFoundError(f"Teacher cache missing for stem={stem} in {self.cache_dir}")

        self.mem[stem] = arr
        if len(self.mem) > self.max_keep:
            self.mem.popitem(last=False)
        return arr

    def get_batch_crop_255(self, metas_list: List[Dict[str, Any]], device):
        crops = []
        for m in metas_list:
            arr = self._load_any(m["stem"])  # HWC
            x, y = int(m["x"]), int(m["y"])
            ps, scale = int(m["ps"]), int(m["scale"])
            xs, ys = x * scale, y * scale
            hs = ps * scale

            crop = arr[ys:ys + hs, xs:xs + hs, :]  # HWC

            if crop.dtype == np.uint8:
                t = torch.from_numpy(crop).to(torch.float32)  # HWC float
            else:
                t = torch.from_numpy(crop.astype(np.float32))

            t = t.permute(2, 0, 1)  # CHW
            crops.append(t)

        return torch.stack(crops, 0).to(device)  # BCHW float32 [0..255]


def _normalize_metas(metas: Any, batch_size: int) -> Optional[List[Dict[str, Any]]]:
    """
    DataLoader collate for dict => dict of lists/tensors.
    Convert to list[dict] length B.
    """
    if metas is None:
        return None

    # Already list of dict
    if isinstance(metas, list) and (len(metas) == 0 or isinstance(metas[0], dict)):
        return metas

    # Collated dict: {"stem": [..], "x": tensor([..]), ...}
    if isinstance(metas, dict):
        out: List[Dict[str, Any]] = []
        for i in range(batch_size):
            mi: Dict[str, Any] = {}
            for k, v in metas.items():
                if isinstance(v, torch.Tensor):
                    mi[k] = int(v[i].item())
                elif isinstance(v, (list, tuple)):
                    mi[k] = v[i]
                else:
                    mi[k] = v
            # stem sometimes comes as list[str] (ok)
            # ensure stem is str
            if isinstance(mi.get("stem"), (list, tuple)):
                mi["stem"] = mi["stem"][0]
            out.append(mi)
        return out

    return None


# -------------------------
# Tricks
# -------------------------
def channel_shuffle_rgb(lr: torch.Tensor, hr: torch.Tensor):
    perm = torch.randperm(3, device=lr.device)
    return lr[:, perm], hr[:, perm]

@torch.no_grad()
def apply_weight_clipping(model: torch.nn.Module, clip_other: float = 2.0, clip_rep: float = 3.0):
    for name, p in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        if p.ndim != 4:
            continue
        lim = clip_rep if "rep." in name else clip_other
        p.clamp_(-lim, lim)

def freeze_bn_(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            if m.weight is not None:
                m.weight.requires_grad_(False)
            if m.bias is not None:
                m.bias.requires_grad_(False)


# -------------------------
# Schedulers
# -------------------------
def make_cosine_warmup_scheduler(opt, total_epochs: int, warmup_ratio: float = 0.1):
    warmup_epochs = max(1, int(total_epochs * warmup_ratio))
    def lr_lambda(ep: int):
        if ep < warmup_epochs:
            return float(ep + 1) / float(warmup_epochs)
        t = (ep - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

def make_step_halve_scheduler(opt, step_size: int = 40, gamma: float = 0.5):
    return torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)


# -------------------------
# QAT helpers
# -------------------------
def prepare_qat(model: torch.nn.Module, backend: str = "qnnpack") -> torch.nn.Module:
    import torch.ao.quantization as tq
    torch.backends.quantized.engine = backend
    model.train()
    model.qconfig = tq.get_default_qat_qconfig(backend)
    tq.prepare_qat(model, inplace=True)
    return model

def strip_qat_to_clean_state(clean_model: torch.nn.Module, qat_state: dict) -> dict:
    clean_sd = clean_model.state_dict()
    return {k: v for k, v in qat_state.items() if (k in clean_sd and v.shape == clean_sd[k].shape)}

def qat_schedule_step(model: torch.nn.Module, local_ep: int,
                      disable_observer_ep: int, freeze_fakequant_ep: int):
    import torch.ao.quantization as tq
    if disable_observer_ep >= 0 and local_ep == disable_observer_ep:
        tq.disable_observer(model)
    if freeze_fakequant_ep >= 0 and local_ep == freeze_fakequant_ep:
        if hasattr(tq, "freeze_fake_quant"):
            tq.freeze_fake_quant(model)
        else:
            tq.disable_observer(model)


# -------------------------
# CKPT helpers
# -------------------------
def load_ckpt_weights(model, ckpt_path, device, prefer_ema: bool = True) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Invalid ckpt: {ckpt_path}")

    if prefer_ema and ("model_ema" in ckpt):
        sd = ckpt["model_ema"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        raise RuntimeError(f"Invalid ckpt keys: {ckpt.keys()}")

    model.load_state_dict(sd, strict=True)
    model.to(device)
    return ckpt

def load_teacher(teacher_ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """
    Optional fallback teacher (AntSR). If you use teacher_cache_dir, you can skip this.
    """
    tckpt = torch.load(teacher_ckpt_path, map_location="cpu")
    if not (isinstance(tckpt, dict) and "cfg" in tckpt and "model" in tckpt):
        raise RuntimeError("Teacher ckpt must be export ckpt with keys {'cfg','model'}.")
    tcfg = tckpt["cfg"]
    tsd = tckpt["model"]
    teacher = AntSR(deploy=True, **tcfg).eval().to(device)
    teacher.load_state_dict(tsd, strict=True)
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


# -------------------------
# Loss builder (on 0..1)
# -------------------------
def compute_base_loss(
    mode: str,
    sr01: torch.Tensor,
    hr01: torch.Tensor,
    dct_w: float,
) -> torch.Tensor:
    mode = mode.lower()
    if mode == "l1":
        return F.l1_loss(sr01, hr01)
    if mode == "l2":
        return F.mse_loss(sr01, hr01)
    if mode == "dct":
        return dct_l1_loss(sr01, hr01)
    if mode == "l1dct":
        return F.l1_loss(sr01, hr01) + dct_w * dct_l1_loss(sr01, hr01)
    if mode == "l2dct":
        return F.mse_loss(sr01, hr01) + dct_w * dct_l1_loss(sr01, hr01)
    raise ValueError(f"Unknown loss_mode: {mode}")


# -------------------------
# Train one stage (EMA + KD + RESUME) with FULL validation
# -------------------------
def train_stage(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int,
    base_lr: float,
    out_dir: str,
    start_epoch: int,
    val_max: int,
    stage_tag: str,

    # full val
    val_shaves: Tuple[int, ...] = (0, 3),
    best_key: str = "psnr_sr_rgb_sh0",
    report_ssim: bool = False,
    ssim_win: int = 11,
    ssim_sigma: float = 1.5,

    grad_clip: float = 0.0,
    scheduler_type: str = "none",
    channel_shuffle: bool = False,

    loss_mode: str = "l1",
    dct_w: float = 0.05,

    weight_clip: bool = True,
    wc_other: float = 2.0,
    wc_rep: float = 3.0,

    val_every: int = 1,

    # EMA
    ema: Optional[EMA] = None,

    # KD
    teacher: Optional[torch.nn.Module] = None,
    kd_w: float = 0.0,
    kd_loss: str = "l1",
    kd_freq_w: float = 0.0,

    # Teacher cache (for KD)
    teacher_cache_dir: Optional[str] = None,

    # BN freeze
    freeze_bn_epoch: int = -1,

    # QAT schedule (stage-local)
    qat_disable_observer_ep: int = -1,
    qat_freeze_fakequant_ep: int = -1,

    # RESUME
    resume_ckpt: Optional[dict] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, f"ckpt_best_{stage_tag}.pt")
    last_path = os.path.join(out_dir, f"ckpt_last_{stage_tag}.pt")

    opt = torch.optim.Adam(model.parameters(), lr=base_lr)

    if scheduler_type == "cos_warmup":
        sch = make_cosine_warmup_scheduler(opt, total_epochs=epochs, warmup_ratio=0.1)
    elif scheduler_type == "step_halve":
        sch = make_step_halve_scheduler(opt, step_size=40, gamma=0.5)
    else:
        sch = None

    # KD cache
    tcache = TeacherCache(teacher_cache_dir) if teacher_cache_dir else None

    # ---- RESUME ----
    local_ep_start = 0
    best_score = -1e9
    bn_frozen = False

    if resume_ckpt is not None:
        if "model" in resume_ckpt:
            model.load_state_dict(resume_ckpt["model"], strict=True)
        elif "model_ema" in resume_ckpt:
            model.load_state_dict(resume_ckpt["model_ema"], strict=True)

        if "opt" in resume_ckpt:
            try:
                opt.load_state_dict(resume_ckpt["opt"])
            except Exception as e:
                print("[resume] WARN: cannot load optimizer state:", e)

        if sch is not None and "sch" in resume_ckpt and resume_ckpt["sch"] is not None:
            try:
                sch.load_state_dict(resume_ckpt["sch"])
            except Exception as e:
                print("[resume] WARN: cannot load scheduler state:", e)

        if ema is not None and "ema" in resume_ckpt and resume_ckpt["ema"] is not None:
            try:
                ema.load_state_dict(resume_ckpt["ema"], model)
            except Exception as e:
                print("[resume] WARN: cannot load EMA shadow:", e)

        best_score = float(resume_ckpt.get("best_score", best_score))
        last_epoch = int(resume_ckpt.get("epoch", start_epoch - 1))
        local_ep_start = max(0, (last_epoch - start_epoch + 1))
        print(f"[resume] stage={stage_tag} last_epoch={last_epoch} -> local_ep_start={local_ep_start}/{epochs}")

    # --------------------
    for local_ep in range(local_ep_start, epochs):
        ep = start_epoch + local_ep

        if (not bn_frozen) and (freeze_bn_epoch >= 0) and (ep >= freeze_bn_epoch):
            freeze_bn_(model)
            bn_frozen = True

        if stage_tag.startswith("s3") and (qat_disable_observer_ep >= 0 or qat_freeze_fakequant_ep >= 0):
            qat_schedule_step(model, local_ep, qat_disable_observer_ep, qat_freeze_fakequant_ep)

        model.train()
        pbar = tqdm(train_loader, desc=f"[{stage_tag}] epoch {ep} lr={opt.param_groups[0]['lr']:.2e}")
        losses = []

        for batch in pbar:
            # batch can be (lr,hr) OR (lr,hr,meta)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                lr_img, hr_img, metas = batch
            else:
                lr_img, hr_img = batch
                metas = None

            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            if channel_shuffle:
                lr_img, hr_img = channel_shuffle_rgb(lr_img, hr_img)

            opt.zero_grad(set_to_none=True)

            sr = model(lr_img)
            sr01 = sr / 255.0
            hr01 = hr_img / 255.0

            base = compute_base_loss(loss_mode, sr01, hr01, dct_w=dct_w)
            loss = base

            # ---- KD ----
            if kd_w > 0:
                t_sr = None

                # 1) Prefer cache if available
                if tcache is not None and metas is not None:
                    metas_list = _normalize_metas(metas, batch_size=lr_img.size(0))
                    if metas_list is None:
                        raise RuntimeError("Cannot normalize metas from DataLoader. Use batch=1 or check collate.")
                    try:
                        t_sr = tcache.get_batch_crop_255(metas_list, device=device)
                    except Exception as e:
                        # fallback to teacher if provided
                        if teacher is None:
                            raise RuntimeError(f"Teacher cache failed and no teacher provided. Error: {e}")
                        t_sr = None

                # 2) Fallback to teacher forward
                if t_sr is None:
                    if teacher is None:
                        raise RuntimeError("KD requested (kd_w>0) but no teacher and no teacher_cache_dir.")
                    with torch.no_grad():
                        t_sr = teacher(lr_img).detach()

                t01 = t_sr / 255.0
                if kd_loss == "l2":
                    kd_pix = F.mse_loss(sr01, t01)
                else:
                    kd_pix = F.l1_loss(sr01, t01)
                loss = loss + kd_w * kd_pix

                if kd_freq_w > 0:
                    loss = loss + kd_freq_w * dct_l1_loss(sr01, t01)

            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            if weight_clip:
                apply_weight_clipping(model, clip_other=wc_other, clip_rep=wc_rep)
            if ema is not None:
                ema.update(model)

            losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(losses)))

        if sch is not None:
            sch.step()

        do_val = (val_every <= 1) or ((local_ep + 1) % val_every == 0) or (local_ep == epochs - 1)
        if do_val:
            if ema is not None:
                ema.apply(model)

            m_all = validate_metrics_all(
                model, val_loader, device,
                scale=3,
                max_images=val_max,
                shaves=val_shaves,
                report_ssim=report_ssim,
                ssim_win=ssim_win,
                ssim_sigma=ssim_sigma,
            )

            if ema is not None:
                ema.restore(model)

            def _g(k: str) -> float:
                return float(m_all.get(k, 0.0))

            msg = (
                f"[val/{stage_tag}] epoch {ep} | "
                f"SR RGB sh0={_g('psnr_sr_rgb_sh0'):.4f}  sh3={_g('psnr_sr_rgb_sh3'):.4f} | "
                f"SR Y   sh0={_g('psnr_sr_y_sh0'):.4f}   sh3={_g('psnr_sr_y_sh3'):.4f} || "
                f"BI RGB sh0={_g('psnr_bi_rgb_sh0'):.4f} sh3={_g('psnr_bi_rgb_sh3'):.4f} | "
                f"BI Y   sh0={_g('psnr_bi_y_sh0'):.4f}  sh3={_g('psnr_bi_y_sh3'):.4f}"
            )
            if report_ssim:
                msg += (
                    f" || SSIM(SR) sh0={_g('ssim_sr_rgb_sh0'):.5f} sh3={_g('ssim_sr_rgb_sh3'):.5f}"
                )
            print(msg)

            cur_score = float(m_all.get(best_key, 0.0))

            ckpt = {
                "epoch": ep,
                "stage": stage_tag,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sch": (sch.state_dict() if sch is not None else None),
                "val_metrics": m_all,
                "best_key": best_key,
                "best_score": float(max(best_score, cur_score)),
                "loss_mode": loss_mode,
                "dct_w": float(dct_w),
                "kd_w": float(kd_w),
                "kd_freq_w": float(kd_freq_w),
                "kd_loss": kd_loss,
                "has_ema": (ema is not None),
                "ema": (ema.state_dict() if ema is not None else None),
            }

            if ema is not None:
                ema.apply(model)
                ckpt["model_ema"] = model.state_dict()
                ema.restore(model)

            torch.save(ckpt, last_path)

            if cur_score > best_score:
                best_score = cur_score
                ckpt["best_score"] = float(best_score)
                torch.save(ckpt, best_path)
                print(f"  -> saved best ({stage_tag}) by {best_key}={best_score:.4f}")

    return best_path


# -------------------------
# Arg helpers
# -------------------------
def _parse_int_list(s: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    vals = [int(p) for p in parts]
    if len(vals) == 0:
        return (0, 3)
    return tuple(vals)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs/antsr_sc")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val_max", type=int, default=0)
    ap.add_argument("--val_every", type=int, default=1)

    ap.add_argument("--resume", type=str, default=None)

    ap.add_argument("--val_shaves", type=str, default="0,3")
    ap.add_argument("--best_key", type=str, default="psnr_sr_rgb_sh0")
    ap.add_argument("--report_ssim", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--ssim_win", type=int, default=11)
    ap.add_argument("--ssim_sigma", type=float, default=1.5)

    ap.add_argument("--shave", type=int, default=3)
    ap.add_argument("--psnr_y", action=argparse.BooleanOptionalAction, default=False)

    # Model core
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--n_rep", type=int, default=4)
    ap.add_argument("--skip_mode", type=str, default="concat_raw",
                    choices=["add", "add1x1", "concat_lr", "concat_raw"])
    ap.add_argument("--concat_htr", type=str, default="3x3_3x3",
                    choices=["3x3_3x3", "1x1_3x3", "1x1_1x1"])
    ap.add_argument("--use_global_add", action=argparse.BooleanOptionalAction, default=True)

    # rep blocks
    ap.add_argument("--rep_type", type=str, default="repconv", choices=["repconv", "mobileone", "repdw"])
    ap.add_argument("--mo_branches", type=int, default=2)
    ap.add_argument("--mo_use_1x1", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mo_use_identity", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--rep_use_bn", action="store_true")
    ap.add_argument("--rep_act_mode", type=str, default="none", choices=["none", "relu"])

    # clamp per phase
    ap.add_argument("--out_clamp_fp32", type=str, default="none",
                    choices=["min255", "minclip", "clamp_0_255", "none"])
    ap.add_argument("--out_clamp_export", type=str, default="minclip",
                    choices=["min255", "minclip", "clamp_0_255", "none"])
    ap.add_argument("--out_clamp_qat", type=str, default="minclip",
                    choices=["min255", "minclip", "clamp_0_255", "none"])

    # Stages
    ap.add_argument("--epochs1", type=int, default=800)
    ap.add_argument("--epochs2", type=int, default=200)
    ap.add_argument("--epochs3", type=int, default=300)

    ap.add_argument("--lr1", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=2e-5)
    ap.add_argument("--lr3", type=float, default=1e-5)

    ap.add_argument("--patch1", type=int, default=128)
    ap.add_argument("--patch2", type=int, default=128)
    ap.add_argument("--patch3", type=int, default=128)

    ap.add_argument("--s1_loss", type=str, default="l1", choices=["l1", "l2", "dct", "l1dct", "l2dct"])
    ap.add_argument("--s2_loss", type=str, default="l2", choices=["l1", "l2", "dct", "l1dct", "l2dct"])
    ap.add_argument("--s3_loss", type=str, default="l1dct", choices=["l1", "l2", "dct", "l1dct", "l2dct"])

    ap.add_argument("--dct_w_s1", type=float, default=0.0)
    ap.add_argument("--dct_w_s2", type=float, default=0.0)
    ap.add_argument("--dct_w_s3", type=float, default=0.05)

    ap.add_argument("--scheduler1", type=str, default="cos_warmup", choices=["none", "cos_warmup"])
    ap.add_argument("--scheduler2", type=str, default="step_halve", choices=["none", "step_halve"])
    ap.add_argument("--scheduler3", type=str, default="step_halve", choices=["none", "step_halve"])

    ap.add_argument("--channel_shuffle_s2s3", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--grad_clip", type=float, default=0.0)

    # EMA + KD
    ap.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ema_decay", type=float, default=0.999)

    # Optional fallback teacher (AntSR). If you use teacher_cache_dir, you can omit teacher_ckpt.
    ap.add_argument("--teacher_ckpt", type=str, default=None)
    ap.add_argument("--teacher_from_stage2", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--kd_w_s1", type=float, default=0.0)
    ap.add_argument("--kd_w_s2", type=float, default=0.0)
    ap.add_argument("--kd_w_s3", type=float, default=0.02)
    ap.add_argument("--kd_freq_w_s3", type=float, default=0.01)
    ap.add_argument("--kd_loss", type=str, default="l1", choices=["l1", "l2"])

    ap.add_argument("--freeze_bn_epoch", type=int, default=80)

    # QAT
    ap.add_argument("--qat", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--qat_backend", type=str, default="qnnpack", choices=["qnnpack", "fbgemm"])
    ap.add_argument("--deploy_before_qat", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--qat_disable_observer_ep", type=int, default=10)
    ap.add_argument("--qat_freeze_fakequant_ep", type=int, default=60)

    # Weight clipping
    ap.add_argument("--weight_clipping", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--wc_other", type=float, default=2.0)
    ap.add_argument("--wc_rep", type=float, default=3.0)

    ap.add_argument("--init_ckpt", type=str, default=None)
    ap.add_argument("--preset", type=str, default="balanced", choices=["balanced", "speed"])

    # Stage3 subset + teacher cache
    ap.add_argument("--stage3_ids_txt", type=str, default=None)
    ap.add_argument("--teacher_cache_dir", type=str, default=None)
    ap.add_argument("--stage3_batch1", action=argparse.BooleanOptionalAction, default=True)

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    if args.preset == "speed":
        args.channels = min(args.channels, 24)
        args.n_rep = max(args.n_rep, 4)
        args.use_global_add = False
        args.skip_mode = "concat_raw"
        args.concat_htr = "1x1_1x1"
    else:
        args.use_global_add = True

    val_shaves = _parse_int_list(args.val_shaves)

    train_hr, train_lr, valid_hr, valid_lr = resolve_div2k_paths(args.data_root, scale=3)

    def make_train_loader(lr_patch: int, id_list_txt=None, return_meta=False, batch_override=None, augment=True):
        train_ds = DIV2KPairX3(
            train_hr, train_lr,
            train=True, lr_patch=lr_patch, augment=augment, repeat=1,
            id_list_txt=id_list_txt,
            return_meta=return_meta
        )
        bs = batch_override if batch_override is not None else args.batch
        return DataLoader(
            train_ds, batch_size=bs, shuffle=True, num_workers=args.workers,
            drop_last=True, pin_memory=(device.type == "cuda")
        )

    val_ds = DIV2KPairX3(valid_hr, valid_lr, train=False, lr_patch=128, augment=False, repeat=1)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.workers,
        pin_memory=(device.type == "cuda")
    )

    cfg = dict(
        scale=3,
        channels=args.channels,
        n_rep=args.n_rep,
        rep_type=args.rep_type,
        rep_use_bn=args.rep_use_bn,
        rep_act_mode=args.rep_act_mode,
        mo_branches=args.mo_branches,
        mo_use_1x1=args.mo_use_1x1,
        mo_use_identity=args.mo_use_identity,
        out_clamp_mode=args.out_clamp_fp32,
        skip_mode=args.skip_mode,
        concat_htr=args.concat_htr,
        use_global_add=args.use_global_add,
    )
    print("MODEL CONFIG:", cfg)
    print("VAL SHAVES:", val_shaves, "BEST KEY:", args.best_key)

    run_dir = os.path.join(args.out_dir, args.preset)
    os.makedirs(run_dir, exist_ok=True)

    model = AntSR(deploy=False, **cfg).to(device)

    teacher = None
    if args.teacher_ckpt:
        teacher = load_teacher(args.teacher_ckpt, device)
        print("Teacher loaded from:", args.teacher_ckpt)

    if args.init_ckpt is not None and args.resume is None:
        print("== Init weights from ckpt ==", args.init_ckpt)
        load_ckpt_weights(model, args.init_ckpt, device, prefer_ema=True)

    resume_ckpt = None
    resume_stage = None
    if args.resume is not None:
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        resume_stage = resume_ckpt.get("stage", None)
        if resume_stage is None:
            raise RuntimeError("Resume ckpt missing key 'stage'.")
        print(f"[resume] loaded {args.resume} stage={resume_stage} epoch={resume_ckpt.get('epoch')}")

    ema = EMA(model, decay=args.ema_decay) if args.ema else None
    if ema is not None and resume_ckpt is not None and resume_ckpt.get("ema") is not None:
        ema.load_state_dict(resume_ckpt["ema"], model)

    # ---- Stage 1 ----
    if args.epochs1 > 0 and (resume_stage is None or resume_stage == "s1_fp32"):
        model.set_out_clamp_mode(args.out_clamp_fp32)
        train_loader = make_train_loader(args.patch1)
        s1_best = train_stage(
            model, train_loader, val_loader, device,
            epochs=args.epochs1, base_lr=args.lr1, out_dir=run_dir,
            start_epoch=0, val_max=args.val_max, stage_tag="s1_fp32",
            val_shaves=val_shaves, best_key=args.best_key,
            report_ssim=args.report_ssim, ssim_win=args.ssim_win, ssim_sigma=args.ssim_sigma,
            grad_clip=args.grad_clip, scheduler_type=args.scheduler1,
            channel_shuffle=False,
            loss_mode=args.s1_loss, dct_w=args.dct_w_s1,
            weight_clip=False, val_every=args.val_every,
            ema=ema,
            teacher=teacher, kd_w=args.kd_w_s1, kd_loss=args.kd_loss, kd_freq_w=0.0,
            teacher_cache_dir=None,
            freeze_bn_epoch=args.freeze_bn_epoch,
            resume_ckpt=(resume_ckpt if resume_stage == "s1_fp32" else None),
        )
        print("Load best Stage1 ->", s1_best)
        load_ckpt_weights(model, s1_best, device, prefer_ema=True)
        ema = EMA(model, decay=args.ema_decay) if args.ema else None

    # ---- Stage 2 ----
    s2_export_path = os.path.join(run_dir, "ckpt_best_s2_deploy.pt")
    if args.epochs2 > 0 and (resume_stage is None or resume_stage == "s2_fp32"):
        model.set_out_clamp_mode(args.out_clamp_fp32)
        train_loader = make_train_loader(args.patch2)
        start_ep = args.epochs1
        s2_best = train_stage(
            model, train_loader, val_loader, device,
            epochs=args.epochs2, base_lr=args.lr2, out_dir=run_dir,
            start_epoch=start_ep, val_max=args.val_max, stage_tag="s2_fp32",
            val_shaves=val_shaves, best_key=args.best_key,
            report_ssim=args.report_ssim, ssim_win=args.ssim_win, ssim_sigma=args.ssim_sigma,
            grad_clip=args.grad_clip, scheduler_type=args.scheduler2,
            channel_shuffle=args.channel_shuffle_s2s3,
            loss_mode=args.s2_loss, dct_w=args.dct_w_s2,
            weight_clip=args.weight_clipping, wc_other=args.wc_other, wc_rep=args.wc_rep,
            val_every=args.val_every,
            ema=ema,
            teacher=teacher, kd_w=args.kd_w_s2, kd_loss=args.kd_loss, kd_freq_w=0.0,
            teacher_cache_dir=None,
            freeze_bn_epoch=args.freeze_bn_epoch,
            resume_ckpt=(resume_ckpt if resume_stage == "s2_fp32" else None),
        )
        print("Load best Stage2 ->", s2_best)
        load_ckpt_weights(model, s2_best, device, prefer_ema=True)

        # Export deploy after stage2
        model.set_out_clamp_mode(args.out_clamp_export)
        model.eval()
        model.switch_to_deploy()
        export_cfg = dict(cfg)
        export_cfg["out_clamp_mode"] = args.out_clamp_export
        torch.save({"model": model.state_dict(), "cfg": export_cfg}, s2_export_path)
        print("Saved:", s2_export_path)

        # rebuild non-deploy for stage3
        model = AntSR(deploy=False, **cfg).to(device)
        load_ckpt_weights(model, s2_best, device, prefer_ema=True)

        if teacher is None and args.teacher_from_stage2:
            teacher = load_teacher(s2_export_path, device)
            print("Teacher auto-loaded from stage2 export:", s2_export_path)

    # ---- Stage 3 (QAT) ----
    if args.qat and args.epochs3 > 0 and (resume_stage is None or resume_stage == "s3_qat"):
        model.set_out_clamp_mode(args.out_clamp_qat)

        # IMPORTANT: stage3 uses subset + meta if teacher_cache_dir is provided
        use_meta = (args.teacher_cache_dir is not None)
        s3_bs = 1 if args.stage3_batch1 else args.batch
        train_loader = make_train_loader(
            args.patch3,
            id_list_txt=args.stage3_ids_txt,
            return_meta=use_meta,
            batch_override=s3_bs,
            augment=False if (args.teacher_cache_dir is not None) else True
        )

        start_ep = args.epochs1 + args.epochs2

        qat_on_deploy = False
        if args.deploy_before_qat:
            model.eval()
            model.switch_to_deploy()
            model.train()
            qat_on_deploy = True

        model = prepare_qat(model, backend=args.qat_backend).to(device)

        ema_qat = EMA(model, decay=args.ema_decay) if args.ema else None
        if ema_qat is not None and resume_ckpt is not None and resume_stage == "s3_qat" and resume_ckpt.get("ema") is not None:
            ema_qat.load_state_dict(resume_ckpt["ema"], model)

        s3_best = train_stage(
            model, train_loader, val_loader, device,
            epochs=args.epochs3, base_lr=args.lr3, out_dir=run_dir,
            start_epoch=start_ep, val_max=args.val_max, stage_tag="s3_qat",
            val_shaves=val_shaves, best_key=args.best_key,
            report_ssim=args.report_ssim, ssim_win=args.ssim_win, ssim_sigma=args.ssim_sigma,
            grad_clip=args.grad_clip, scheduler_type=args.scheduler3,
            channel_shuffle=False if (args.teacher_cache_dir is not None or args.kd_w_s3 > 0) else args.channel_shuffle_s2s3,
            loss_mode=args.s3_loss, dct_w=args.dct_w_s3,
            weight_clip=args.weight_clipping, wc_other=args.wc_other, wc_rep=args.wc_rep,
            val_every=args.val_every,
            ema=ema_qat,
            teacher=teacher,                 # fallback only
            kd_w=args.kd_w_s3,
            kd_loss=args.kd_loss,
            kd_freq_w=args.kd_freq_w_s3,
            teacher_cache_dir=args.teacher_cache_dir,  # <<< KD from cache
            freeze_bn_epoch=args.freeze_bn_epoch,
            qat_disable_observer_ep=args.qat_disable_observer_ep,
            qat_freeze_fakequant_ep=args.qat_freeze_fakequant_ep,
            resume_ckpt=(resume_ckpt if resume_stage == "s3_qat" else None),
        )
        print("Best QAT ckpt:", s3_best)

        qat_ckpt = torch.load(s3_best, map_location="cpu")
        qat_state = qat_ckpt.get("model_ema", qat_ckpt["model"])

        clean_cfg = dict(cfg)
        clean_cfg["out_clamp_mode"] = args.out_clamp_export

        clean = AntSR(deploy=qat_on_deploy, **clean_cfg).eval()
        filtered = strip_qat_to_clean_state(clean, qat_state)
        clean.load_state_dict(filtered, strict=False)

        clean.set_out_clamp_mode(args.out_clamp_export)
        clean.switch_to_deploy()

        deploy_path = os.path.join(run_dir, "ckpt_best_s3_qat_deploy.pt")
        torch.save({"model": clean.state_dict(), "cfg": clean_cfg}, deploy_path)
        print("Saved:", deploy_path)

    print("Done. run_dir:", run_dir)


if __name__ == "__main__":
    main()