import os
import glob
import math
import argparse
from typing import Dict, Tuple

import numpy as np
from PIL import Image


def read_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)


def rgb_to_y(x: np.ndarray) -> np.ndarray:
    # x: HWC, 0..255
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def shave(x: np.ndarray, s: int) -> np.ndarray:
    if s <= 0:
        return x
    h, w = x.shape[:2]
    if h <= 2 * s or w <= 2 * s:
        return x
    return x[s:-s, s:-s, ...]


def psnr_255(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse < eps:
        return 99.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def find_div2k_valid_dirs(data_root: str) -> tuple[str, str]:
    hr_cands = [
        os.path.join(data_root, "DIV2K_valid_HR"),
        os.path.join(data_root, "DIV2K_valid_HR", "DIV2K_valid_HR"),
    ]
    lr_cands = [
        os.path.join(data_root, "DIV2K_valid_LR_bicubic", "X3"),
        os.path.join(data_root, "DIV2K_valid_LR_bicubic_X3"),
        os.path.join(data_root, "DIV2K_valid_LR_bicubic_X3", "DIV2K_valid_LR_bicubic", "X3"),
    ]

    hr_dir = next((d for d in hr_cands if os.path.isdir(d)), None)
    lr_dir = next((d for d in lr_cands if os.path.isdir(d)), None)

    if hr_dir is None or lr_dir is None:
        raise FileNotFoundError("Cannot resolve DIV2K valid HR/LR x3 dirs.")

    if not glob.glob(os.path.join(hr_dir, "*.png")):
        nested = os.path.join(hr_dir, os.path.basename(hr_dir))
        if os.path.isdir(nested) and glob.glob(os.path.join(nested, "*.png")):
            hr_dir = nested

    if not glob.glob(os.path.join(lr_dir, "*.png")):
        nested_lr = os.path.join(lr_dir, "DIV2K_valid_LR_bicubic", "X3")
        if os.path.isdir(nested_lr) and glob.glob(os.path.join(nested_lr, "*.png")):
            lr_dir = nested_lr

    return hr_dir, lr_dir


def load_pairs(data_root: str):
    hr_dir, lr_dir = find_div2k_valid_dirs(data_root)
    hr_files = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
    if not hr_files:
        raise FileNotFoundError(f"No HR png found in: {hr_dir}")

    pairs = []
    for hp in hr_files:
        stem = os.path.splitext(os.path.basename(hp))[0]
        lp = os.path.join(lr_dir, f"{stem}x3.png")
        if not os.path.isfile(lp):
            lp = os.path.join(lr_dir, f"{stem}.png")
        if not os.path.isfile(lp):
            raise FileNotFoundError(f"Missing LR for {stem}")
        pairs.append((lp, hp, stem))
    return pairs


def get_interpreter(tflite_path: str):
    try:
        from ai_edge_litert.interpreter import Interpreter
        itp = Interpreter(model_path=tflite_path)
    except Exception:
        import tensorflow as tf
        itp = tf.lite.Interpreter(model_path=tflite_path)

    itp.allocate_tensors()
    return itp


def preprocess_input(img_hwc_255: np.ndarray, detail: dict) -> np.ndarray:
    shape = tuple(int(x) for x in detail["shape"])
    dtype = detail["dtype"]
    scale, zero_point = detail.get("quantization", (0.0, 0))

    if len(shape) != 4 or shape[0] != 1:
        raise RuntimeError(f"Unsupported input shape: {shape}")

    if shape[1] in (1, 3):  # NCHW
        h, w = shape[2], shape[3]
        x = np.array(Image.fromarray(img_hwc_255.astype(np.uint8)).resize((w, h), Image.BICUBIC),
                     dtype=np.float32)
        x = np.transpose(x, (2, 0, 1))[None, ...]
    elif shape[3] in (1, 3):  # NHWC
        h, w = shape[1], shape[2]
        x = np.array(Image.fromarray(img_hwc_255.astype(np.uint8)).resize((w, h), Image.BICUBIC),
                     dtype=np.float32)[None, ...]
    else:
        raise RuntimeError(f"Cannot infer input layout from shape={shape}")

    if np.issubdtype(dtype, np.integer):
        if scale <= 0:
            raise RuntimeError(f"Invalid input quantization params: {detail.get('quantization')}")
        x = np.round(x / scale + zero_point).astype(dtype)
    else:
        x = x.astype(dtype)

    return x


def postprocess_output(y: np.ndarray, detail: dict) -> np.ndarray:
    dtype = detail["dtype"]
    scale, zero_point = detail.get("quantization", (0.0, 0))

    if np.issubdtype(dtype, np.integer):
        if scale <= 0:
            raise RuntimeError(f"Invalid output quantization params: {detail.get('quantization')}")
        y = (y.astype(np.float32) - zero_point) * scale
    else:
        y = y.astype(np.float32)

    if y.ndim != 4 or y.shape[0] != 1:
        raise RuntimeError(f"Unsupported output shape: {y.shape}")

    # NCHW or NHWC -> HWC
    if y.shape[1] in (1, 3):
        y = np.transpose(y[0], (1, 2, 0))
    elif y.shape[3] in (1, 3):
        y = y[0]
    else:
        raise RuntimeError(f"Cannot infer output layout from shape={y.shape}")

    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--max_images", type=int, default=0)
    args = ap.parse_args()

    pairs = load_pairs(args.data_root)
    if args.max_images > 0:
        pairs = pairs[:args.max_images]

    itp = get_interpreter(args.tflite)
    in_detail = itp.get_input_details()[0]
    out_detail = itp.get_output_details()[0]

    psnr_rgb_sh0 = []
    psnr_rgb_sh3 = []
    psnr_y_sh0 = []
    psnr_y_sh3 = []

    for lp, hp, stem in pairs:
        lr = read_rgb(lp)
        hr = read_rgb(hp)

        x = preprocess_input(lr, in_detail)
        itp.set_tensor(in_detail["index"], x)
        itp.invoke()
        y = itp.get_tensor(out_detail["index"])
        sr = postprocess_output(y, out_detail)

        H, W = hr.shape[:2]
        sr = sr[:H, :W, :]
        hr = hr[:H, :W, :]

        sr = np.clip(sr, 0.0, 255.0)
        hr = np.clip(hr, 0.0, 255.0)

        for sh, arr_rgb, arr_y in [
            (0, psnr_rgb_sh0, psnr_y_sh0),
            (3, psnr_rgb_sh3, psnr_y_sh3),
        ]:
            sr_s = shave(sr, sh)
            hr_s = shave(hr, sh)
            arr_rgb.append(psnr_255(sr_s, hr_s))
            arr_y.append(psnr_255(rgb_to_y(sr_s), rgb_to_y(hr_s)))

    print(f"num_images={len(pairs)}")
    print(f"psnr_sr_rgb_sh0={np.mean(psnr_rgb_sh0):.6f}")
    print(f"psnr_sr_rgb_sh3={np.mean(psnr_rgb_sh3):.6f}")
    print(f"psnr_sr_y_sh0={np.mean(psnr_y_sh0):.6f}")
    print(f"psnr_sr_y_sh3={np.mean(psnr_y_sh3):.6f}")


if __name__ == "__main__":
    main()