import os
import argparse
import torch
import numpy as np
import torch.nn.functional as F

from model_antsr import AntSR
from data_div2k_pairs import DIV2KPairX3, resolve_div2k_paths

# ----- metric (same as train) -----
def psnr_255(pred: torch.Tensor, gt: torch.Tensor, eps=1e-12) -> float:
    mse = torch.mean((pred - gt) ** 2).item()
    if mse < eps:
        return 99.0
    return 10.0 * np.log10((255.0 * 255.0) / mse)

def _shave_border(a: torch.Tensor, shave: int) -> torch.Tensor:
    if shave <= 0:
        return a
    h, w = a.shape[-2], a.shape[-1]
    if h <= 2 * shave or w <= 2 * shave:
        return a
    return a[..., shave:-shave, shave:-shave]

@torch.no_grad()
def eval_div2k(model, loader, device, shave: int = 3):
    model.eval()
    psnr_sr = []
    psnr_bi = []

    for lr, hr in loader:
        lr = lr.to(device)
        hr = hr.to(device)

        bi = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
        sr = model(lr)

        H, W = hr.shape[-2], hr.shape[-1]
        sr = sr[..., :H, :W]
        bi = bi[..., :H, :W]

        sr = _shave_border(sr, shave)
        bi = _shave_border(bi, shave)
        hr = _shave_border(hr, shave)

        sr = torch.clamp(sr, 0.0, 255.0)
        bi = torch.clamp(bi, 0.0, 255.0)
        hr = torch.clamp(hr, 0.0, 255.0)

        psnr_sr.append(psnr_255(sr, hr))
        psnr_bi.append(psnr_255(bi, hr))

    return {
        "psnr_sr": float(np.mean(psnr_sr)) if psnr_sr else 0.0,
        "psnr_bicubic": float(np.mean(psnr_bi)) if psnr_bi else 0.0,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--prefer_ema", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--shave", type=int, default=3)
    ap.add_argument("--max_images", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    blob = torch.load(args.ckpt, map_location="cpu")
    if not isinstance(blob, dict):
        raise RuntimeError("Unknown ckpt format (not a dict).")

    # -------- pick cfg + state_dict depending on ckpt type --------
    if "cfg" in blob and "model" in blob:
        # export/deploy ckpt
        cfg = blob["cfg"]
        sd = blob["model"]
    elif "model" in blob:
        # training ckpt: need cfg from args? -> infer minimal safe defaults
        # NOTE: better if your training ckpt also saves cfg. If not, you must supply matching args here.
        raise RuntimeError(
            "This is a training ckpt but it does NOT contain 'cfg'.\n"
            "Fix: save cfg into training ckpt, or evaluate the exported deploy ckpt (ckpt_best_s2_deploy / ckpt_best_s3_qat_deploy)."
        )
    else:
        raise RuntimeError(f"Invalid ckpt keys: {list(blob.keys())}")

    model = AntSR(deploy=True, **cfg).to(device).eval()
    model.load_state_dict(sd, strict=True)

    # dataset
    _, _, valid_hr, valid_lr = resolve_div2k_paths(args.data_root, scale=3)
    val_ds = DIV2KPairX3(valid_hr, valid_lr, train=False, lr_patch=128, augment=False, repeat=1)
    if args.max_images > 0:
        # quick subset: slice by wrapping
        from torch.utils.data import Subset
        val_ds = Subset(val_ds, list(range(min(args.max_images, len(val_ds)))))

    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    out = eval_div2k(model, val_loader, device, shave=args.shave)
    print(out)

if __name__ == "__main__":
    main()
