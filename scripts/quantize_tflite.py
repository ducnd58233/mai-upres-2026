from __future__ import annotations

import argparse
import glob
import logging
import random
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from configs import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_rgb_float(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32)


def _make_patch(arr_hwc: np.ndarray, height: int, width: int, nchw: bool) -> np.ndarray:
    h, w = arr_hwc.shape[0], arr_hwc.shape[1]
    y = 0 if h <= height else random.randint(0, h - height)
    x = 0 if w <= width else random.randint(0, w - width)
    patch = arr_hwc[y : y + height, x : x + width, :]
    if patch.shape[0] != height or patch.shape[1] != width:
        pad_h = max(0, height - patch.shape[0])
        pad_w = max(0, width - patch.shape[1])
        patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    if nchw:
        return np.transpose(patch, (2, 0, 1))[None, ...].astype(np.float32)
    return patch[None, ...].astype(np.float32)


def _get_input_spec_from_saved_model(
    saved_model_dir: str,
) -> tuple[tuple[int, ...], bool]:
    import tensorflow as tf

    loaded = tf.saved_model.load(saved_model_dir)
    sigs = getattr(loaded, "signatures", None)
    if not sigs:
        raise RuntimeError(
            "SavedModel has no signatures. Use a model with serving_default or similar."
        )
    sig = sigs.get("serving_default") or next(iter(sigs.values()))
    in_name = next(iter(sig.structured_input_signature[0].keys()))
    # Get shape from the concrete function
    for inp in sig.inputs:
        if inp.name == in_name or inp.name.endswith(in_name):
            shape = tuple(inp.shape.as_list())
            break
    else:
        # Fallback: first input
        inp = sig.inputs[0]
        shape = tuple(inp.shape.as_list())
    # shape is e.g. (1, H, W, 3) or (1, 3, H, W)
    if len(shape) != 4:
        raise ValueError(f"Expected 4D input shape, got {shape}")
    if shape[-1] == 3:
        nchw = False
        _, h, w, _ = shape
    elif shape[1] == 3:
        nchw = True
        _, _, h, w = shape
    else:
        raise ValueError(f"Cannot infer H,W from shape {shape}")
    return (int(h), int(w)), nchw


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Quantize float SavedModel/Keras to INT8 TFLite (TensorFlow, works on Windows).",
    )
    ap.add_argument(
        "--saved-model",
        help="Path to TensorFlow SavedModel directory.",
    )
    ap.add_argument(
        "--keras-model",
        help="Path to .keras file (use instead of --saved-model).",
    )
    ap.add_argument(
        "--out-tflite",
        type=Path,
        default=_PROJECT_ROOT / "runs" / "models" / "export" / "model.tflite",
        help="Output INT8 .tflite path (default: runs/models/export/model.tflite).",
    )
    ap.add_argument(
        "--calib-dir",
        required=True,
        help="Directory of calibration images (e.g. DIV2K LR folder).",
    )
    ap.add_argument(
        "--num-calib", type=int, default=200, help="Number of calibration images."
    )
    ap.add_argument("--glob", default="*.png", help="Glob for calibration files.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if bool(args.saved_model) == bool(args.keras_model):
        ap.error("Provide exactly one of --saved-model or --keras-model.")

    import tensorflow as tf

    if args.keras_model:
        in_path = Path(args.keras_model)
        if not in_path.exists():
            raise FileNotFoundError(f"Keras model not found: {in_path}")
        logger.info("Loading Keras model: %s", args.keras_model)
        keras = getattr(tf, "keras")
        model = keras.models.load_model(args.keras_model)
        # Infer input shape from model
        inp = model.input
        shape = tuple(inp.shape.as_list())
        if len(shape) != 4:
            raise ValueError(f"Expected 4D input, got {shape}")
        if shape[-1] == 3:
            nchw = False
            h, w = int(shape[1]), int(shape[2])
        else:
            nchw = True
            h, w = int(shape[2]), int(shape[3])
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        saved_dir = Path(args.saved_model)
        if not saved_dir.is_dir():
            raise FileNotFoundError(f"SavedModel dir not found: {saved_dir}")
        (h, w), nchw = _get_input_spec_from_saved_model(str(saved_dir))
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))

    calib_dir = Path(args.calib_dir)
    paths = sorted(glob.glob(str(calib_dir / args.glob)))
    if not paths:
        raise FileNotFoundError(f"No files in {calib_dir} matching {args.glob}")

    random.seed(args.seed)
    random.shuffle(paths)
    paths = paths[: max(args.num_calib, 1)]

    def representative_dataset_gen():
        for p in paths:
            arr = _load_rgb_float(p)
            patch = _make_patch(arr, h, w, nchw)
            yield [patch]

    conv = cast(Any, converter)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset_gen
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8

    out_path = Path(args.out_tflite)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Quantizing -> %s (calib: %d images, input %dx%d)",
        out_path,
        len(paths),
        h,
        w,
    )
    tflite_quant = conv.convert()
    if not isinstance(tflite_quant, bytes):
        raise RuntimeError("TFLiteConverter.convert() did not return bytes")
    out_path.write_bytes(tflite_quant)
    logger.info("Saved INT8: %s", out_path)


if __name__ == "__main__":
    main()
