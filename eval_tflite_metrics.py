#!/usr/bin/env python3
import argparse, glob, random, math, os
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

import torch
import torch.nn.functional as F

# ---------- helpers ----------
def load_rgb_255(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)  # HWC [0..255]

def rgb_to_y_np(x: np.ndarray) -> np.ndarray:
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b

def shave_border_np(a: np.ndarray, shave: int) -> np.ndarray:
    if shave <= 0:
        return a
    h, w = a.shape[0], a.shape[1]
    if h <= 2 * shave or w <= 2 * shave:
        return a
    return a[shave:-shave, shave:-shave, :]

def psnr_255_np(a: np.ndarray, b: np.ndarray, eps=1e-12) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse < eps:
        return 99.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)

def quantize_input(x: np.ndarray, dtype, qinfo):
    scale, zp = qinfo
    if scale == 0:
        return x.astype(dtype)
    q = np.round(x / float(scale) + float(zp))
    if dtype == np.int8:
        q = np.clip(q, -128, 127)
    elif dtype == np.uint8:
        q = np.clip(q, 0, 255)
    return q.astype(dtype)

def dequant_output(xq: np.ndarray, qinfo):
    scale, zp = qinfo
    if scale == 0:
        return xq.astype(np.float32)
    return (xq.astype(np.float32) - float(zp)) * float(scale)

def aspect_center_crop(img: Image.Image, target_ratio: float):
    w, h = img.size
    cur = w / h
    if abs(cur - target_ratio) < 1e-6:
        return img
    if cur > target_ratio:
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        return img.crop((x0, 0, x0 + new_w, h))
    else:
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        return img.crop((0, y0, w, y0 + new_h))

def crop_pair(lr: np.ndarray, hr: np.ndarray, H: int, W: int, scale: int = 3,
              allow_resize_small: bool = True, aspect_crop: bool = False):
    # lr: Hlr x Wlr x 3 ; hr: Hhr x Whr x 3
    h, w = lr.shape[:2]
    if h < H or w < W:
        if not allow_resize_small:
            return None, None
        lr_img = Image.fromarray(np.clip(lr, 0, 255).astype(np.uint8))
        hr_img = Image.fromarray(np.clip(hr, 0, 255).astype(np.uint8))
        if aspect_crop:
            lr_img = aspect_center_crop(lr_img, W / H)
            hr_img = aspect_center_crop(hr_img, (W*scale) / (H*scale))
        lr_img = lr_img.resize((W, H), Image.BICUBIC)
        hr_img = hr_img.resize((W*scale, H*scale), Image.BICUBIC)
        return np.array(lr_img, np.float32), np.array(hr_img, np.float32)

    y = 0 if h == H else random.randint(0, h - H)
    x = 0 if w == W else random.randint(0, w - W)
    lr_c = lr[y:y+H, x:x+W, :]

    hr_y, hr_x = y * scale, x * scale
    hr_c = hr[hr_y:hr_y + H*scale, hr_x:hr_x + W*scale, :]
    return lr_c.astype(np.float32), hr_c.astype(np.float32)

# ---------- fast SSIM using torch (CPU) ----------
def _gaussian_1d(win: int, sigma: float, device, dtype):
    coords = torch.arange(win, device=device, dtype=dtype) - (win - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    return g

def _ssim_torch(img1: torch.Tensor, img2: torch.Tensor, win: int = 11, sigma: float = 1.5) -> torch.Tensor:
    # img1,img2: NCHW float32 in [0..255]
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
    return ssim_map.mean(dim=(-1, -2))  # N,C

def _to_nhwc(x: np.ndarray, layout: str) -> np.ndarray:
    # x: with batch
    if layout == "NHWC":
        return x
    # NCHW -> NHWC
    return np.transpose(x, (0, 2, 3, 1))

def _to_hwc(y: np.ndarray, layout: str) -> np.ndarray:
    # y without batch
    if layout == "NHWC":
        return y
    return np.transpose(y, (1, 2, 0))

def infer_layout_from_shape(shape) -> str:
    shape = list(shape)
    if len(shape) != 4:
        return "NHWC"
    if shape[-1] == 3:
        return "NHWC"
    if shape[1] == 3:
        return "NCHW"
    # fallback
    return "NHWC"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lr_dir", required=True)
    ap.add_argument("--hr_dir", required=True)
    ap.add_argument("--num_images", type=int, default=20)
    ap.add_argument("--crops_per_image", type=int, default=1)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--allow_resize_small", action="store_true")
    ap.add_argument("--aspect_crop", action="store_true")
    ap.add_argument("--compute_ssim", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_grad_enabled(False)

    itp = tflite.Interpreter(model_path=args.model, num_threads=args.threads)
    itp.allocate_tensors()
    in0 = itp.get_input_details()[0]
    out0 = itp.get_output_details()[0]

    in_layout = infer_layout_from_shape(in0["shape"])
    out_layout = infer_layout_from_shape(out0["shape"])

    in_dtype = in0["dtype"]
    out_dtype = out0["dtype"]
    in_q = in0.get("quantization", (0.0, 0))
    out_q = out0.get("quantization", (0.0, 0))
    in_index = in0["index"]
    out_index = out0["index"]

    hr_paths = sorted(glob.glob(os.path.join(args.hr_dir, "*.png")))
    if not hr_paths:
        raise FileNotFoundError("No HR pngs found")
    random.shuffle(hr_paths)
    hr_paths = hr_paths[:max(1, args.num_images)]

    shaves = [0, 3]
    acc = {f"psnr_rgb_sh{s}": [] for s in shaves}
    acc.update({f"psnr_y_sh{s}": [] for s in shaves})
    if args.compute_ssim:
        acc.update({f"ssim_rgb_sh{s}": [] for s in shaves})
        acc.update({f"ssim_y_sh{s}": [] for s in shaves})

    for hr_p in hr_paths:
        stem = os.path.splitext(os.path.basename(hr_p))[0]
        lr_p = os.path.join(args.lr_dir, f"{stem}x{args.scale}.png")
        if not os.path.exists(lr_p):
            continue

        hr = load_rgb_255(hr_p)
        lr = load_rgb_255(lr_p)

        for _ in range(max(1, args.crops_per_image)):
            lr_c, hr_c = crop_pair(
                lr, hr, args.h, args.w, scale=args.scale,
                allow_resize_small=args.allow_resize_small,
                aspect_crop=args.aspect_crop
            )
            if lr_c is None:
                continue

            # input batch
            inp = lr_c[None, ...]  # NHWC float
            if in_layout == "NCHW":
                inp = np.transpose(inp, (0, 3, 1, 2))

            if in_dtype != np.float32:
                inp_q = quantize_input(inp, in_dtype, in_q)
            else:
                inp_q = inp.astype(np.float32)

            itp.set_tensor(in_index, inp_q)
            itp.invoke()
            out = itp.get_tensor(out_index)

            if out_dtype != np.float32:
                sr = dequant_output(out, out_q)
            else:
                sr = out.astype(np.float32)

            # to HWC
            sr_hwc = _to_hwc(sr[0], out_layout)
            gt = hr_c

            # align + clamp
            Hh, Wh = gt.shape[0], gt.shape[1]
            sr_hwc = sr_hwc[:Hh, :Wh, :]
            sr_hwc = np.clip(sr_hwc, 0.0, 255.0).astype(np.float32)
            gt = np.clip(gt, 0.0, 255.0).astype(np.float32)

            for sh in shaves:
                sr_s = shave_border_np(sr_hwc, sh)
                gt_s = shave_border_np(gt, sh)

                acc[f"psnr_rgb_sh{sh}"].append(psnr_255_np(sr_s, gt_s))

                sr_y = rgb_to_y_np(sr_s)
                gt_y = rgb_to_y_np(gt_s)
                acc[f"psnr_y_sh{sh}"].append(psnr_255_np(sr_y, gt_y))

                if args.compute_ssim:
                    # torch SSIM (fast)
                    t1 = torch.from_numpy(np.transpose(sr_s, (2,0,1))[None, ...])
                    t2 = torch.from_numpy(np.transpose(gt_s, (2,0,1))[None, ...])
                    ssim_rgb = _ssim_torch(t1, t2).mean().item()
                    acc[f"ssim_rgb_sh{sh}"].append(ssim_rgb)

                    y1 = torch.from_numpy(sr_y[None, None, ...])
                    y2 = torch.from_numpy(gt_y[None, None, ...])
                    ssim_y = _ssim_torch(y1, y2).mean().item()
                    acc[f"ssim_y_sh{sh}"].append(ssim_y)

    print("Model:", args.model)
    print("Input:", in0["shape"], in_layout, in_dtype, in_q)
    print("Output:", out0["shape"], out_layout, out_dtype, out_q)
    for k, v in acc.items():
        if len(v):
            print(f"{k}: {float(np.mean(v)):.4f} (n={len(v)})")
        else:
            print(f"{k}: n=0")

if __name__ == "__main__":
    main()