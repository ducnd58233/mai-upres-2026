from __future__ import annotations

import argparse
import glob
import logging
import math
import os
import time

import numpy as np
from PIL import Image
from tensorflow.lite.python.interpreter import Interpreter

from configs import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Generic TFLite PSNR eval (any .tflite SR model, LR/HR dirs)
# -----------------------------------------------------------------------------


def _psnr_numpy(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse < eps:
        return 99.0
    return 10.0 * math.log10((255.0**2) / mse)


def _map_lr_to_hr_name(lr_name: str, scale: int) -> str:
    stem, ext = os.path.splitext(lr_name)
    suf = f"x{scale}"
    if stem.endswith(suf):
        stem = stem[: -len(suf)]
    return stem + ext


def _pad_reflect_hwc(img: np.ndarray, target_h: int, target_w: int):
    h, w = img.shape[:2]
    if h > target_h or w > target_w:
        y0 = max(0, (h - target_h) // 2)
        x0 = max(0, (w - target_w) // 2)
        img = img[y0 : y0 + target_h, x0 : x0 + target_w, :]
        h, w = img.shape[:2]
    pad_h = target_h - h
    pad_w = target_w - w
    if pad_h == 0 and pad_w == 0:
        return img, 0, 0
    out = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return out, pad_h, pad_w


def _infer_one_tflite(
    interpreter,
    lr_u8_hwc: np.ndarray,
    in_detail: dict,
    out_detail: dict,
    scale: int,
) -> tuple:
    """Run one inference; return (pred_hwc_uint8, pad_h, pad_w)."""
    in_shape = in_detail["shape"]
    in_dtype = in_detail["dtype"]
    in_quant = in_detail.get("quantization", (1.0, 0))
    in_scale, in_zp = in_quant[0], in_quant[1] if len(in_quant) > 1 else 0

    if in_shape[1] == 3:
        Hm, Wm = int(in_shape[2]), int(in_shape[3])
        nchw = True
    else:
        Hm, Wm = int(in_shape[1]), int(in_shape[2])
        nchw = False

    lr_pad, pad_h, pad_w = _pad_reflect_hwc(lr_u8_hwc, Hm, Wm)
    if nchw:
        x = np.transpose(lr_pad, (2, 0, 1))[None, ...].astype(np.float32)
    else:
        x = lr_pad[None, ...].astype(np.float32)

    if in_dtype == np.float32:
        interpreter.set_tensor(in_detail["index"], x)
    else:
        if in_dtype == np.int8:
            xq = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
        else:
            xq = x.astype(np.uint8)
        interpreter.set_tensor(in_detail["index"], xq)

    interpreter.invoke()
    out = interpreter.get_tensor(out_detail["index"])
    out_dtype = out_detail["dtype"]
    out_quant = out_detail.get("quantization", (1.0, 0))
    out_scale, out_zp = out_quant[0], out_quant[1] if len(out_quant) > 1 else 0

    if out_dtype == np.float32:
        out_u8 = np.clip(np.round(out), 0, 255).astype(np.uint8)
    elif out_dtype == np.int8:
        out_u8 = np.clip(
            np.round((out.astype(np.float32) - out_zp) * out_scale), 0, 255
        ).astype(np.uint8)
    else:
        out_u8 = out.astype(np.uint8)

    if out_u8.shape[1] == 3:
        pred_hwc = np.transpose(out_u8[0], (1, 2, 0))
    else:
        pred_hwc = out_u8[0]
    return pred_hwc, pad_h, pad_w


def run_eval_tflite(
    tflite_path: str,
    lr_dir: str,
    hr_dir: str,
    scale: int = 3,
    num_images: int = 0,
    glob_pattern: str = "*.png",
    num_threads: int = 4,
) -> float:
    """Run TFLite model on LR/HR pairs; return mean PSNR (dB)."""
    interp = Interpreter(model_path=tflite_path, num_threads=num_threads)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    lr_files = sorted(glob.glob(os.path.join(lr_dir, glob_pattern)))
    if not lr_files:
        raise FileNotFoundError(f"No files: {os.path.join(lr_dir, glob_pattern)}")
    if num_images > 0:
        lr_files = lr_files[:num_images]

    scores = []
    t0 = time.time()
    for lr_path in lr_files:
        lr_name = os.path.basename(lr_path)
        hr_name = _map_lr_to_hr_name(lr_name, scale)
        hr_path = os.path.join(hr_dir, hr_name)
        if not os.path.exists(hr_path):
            raise FileNotFoundError(f"Missing GT: {hr_path}")

        lr = np.array(Image.open(lr_path).convert("RGB"), dtype=np.uint8)
        gt = np.array(Image.open(hr_path).convert("RGB"), dtype=np.uint8)

        pred, pad_h, pad_w = _infer_one_tflite(interp, lr, in_det, out_det, scale)
        if pad_h > 0 or pad_w > 0:
            H_keep = pred.shape[0] - pad_h * scale
            W_keep = pred.shape[1] - pad_w * scale
            pred = pred[:H_keep, :W_keep, :]
        pred = pred[: gt.shape[0], : gt.shape[1], :]
        if pred.shape != gt.shape:
            pred = pred[: gt.shape[0], : gt.shape[1], :]

        scores.append(_psnr_numpy(pred, gt))

    elapsed = time.time() - t0
    mean_psnr = float(np.mean(scores))
    logger.info(
        "TFLite eval: images=%d PSNR=%.4f dB time=%.2fs",
        len(scores),
        mean_psnr,
        elapsed,
    )
    return mean_psnr


# -----------------------------------------------------------------------------
# CLI and dispatch
# -----------------------------------------------------------------------------

DEFAULT_DATA_ROOT = "datasets/DIV2K"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Eval SR models: PyTorch checkpoint or TFLite on DIV2K.",
    )
    p.add_argument(
        "--model", required=True, choices=["antsr"], help="Model to evaluate."
    )
    p.add_argument(
        "--mode",
        required=True,
        choices=["pytorch", "tflite"],
        help="pytorch: ckpt on DIV2K valid; tflite: .tflite on LR/HR dirs.",
    )
    p.add_argument(
        "--data-root", default=DEFAULT_DATA_ROOT, help="DIV2K root (pytorch mode)."
    )
    p.add_argument("--checkpoint", help="Path to PyTorch checkpoint (pytorch mode).")
    p.add_argument("--tflite", help="Path to .tflite model (tflite mode).")
    p.add_argument("--lr-dir", help="LR images dir (tflite mode).")
    p.add_argument("--hr-dir", help="HR GT images dir (tflite mode).")
    p.add_argument("--scale", type=int, default=3, help="Upscale factor (tflite mode).")
    p.add_argument("--num", type=int, default=0, help="Max images (0=all).")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--batch-size", type=int, default=1, help="Batch size (pytorch mode)."
    )
    p.add_argument("--workers", type=int, default=0)
    p.add_argument(
        "--threads", type=int, default=4, help="TFLite threads (tflite mode)."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "pytorch":
        if not args.checkpoint:
            raise SystemExit("--checkpoint required for --mode pytorch")
        import torch

        from models.antsr.eval import run_antsr_eval_pytorch

        device = torch.device(
            "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        logger.info("Device: %s", device)
        run_antsr_eval_pytorch(
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            device=device,
            max_images=args.num if args.num > 0 else 0,
            batch_size=args.batch_size,
            workers=args.workers,
        )
    elif args.mode == "tflite":
        if not args.tflite or not args.lr_dir or not args.hr_dir:
            raise SystemExit("--tflite, --lr-dir, --hr-dir required for --mode tflite")
        run_eval_tflite(
            tflite_path=args.tflite,
            lr_dir=args.lr_dir,
            hr_dir=args.hr_dir,
            scale=args.scale,
            num_images=args.num if args.num > 0 else 0,
            num_threads=args.threads,
        )
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
