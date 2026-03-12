import os
import glob
import argparse
from typing import Generator, Iterable, Tuple

import numpy as np
from PIL import Image


def read_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)


def resize_hw(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    pil = pil.resize((out_w, out_h), Image.BICUBIC)
    return np.array(pil, dtype=np.float32)


def detect_div2k_lr_dir(data_root: str) -> str:
    patterns = [
        os.path.join(data_root, "DIV2K_valid_LR_bicubic", "X3", "*.png"),
        os.path.join(data_root, "DIV2K_valid_LR_bicubic_X3", "*.png"),
        os.path.join(
            data_root,
            "DIV2K_valid_LR_bicubic_X3",
            "DIV2K_valid_LR_bicubic",
            "X3",
            "*.png",
        ),
        os.path.join(data_root, "DIV2K_train_LR_bicubic", "X3", "*.png"),
        os.path.join(data_root, "DIV2K_train_LR_bicubic_X3", "*.png"),
        os.path.join(
            data_root,
            "DIV2K_train_LR_bicubic_X3",
            "DIV2K_train_LR_bicubic",
            "X3",
            "*.png",
        ),
    ]

    for pat in patterns:
        files = glob.glob(pat)
        if files:
            return os.path.dirname(pat)

    # Fallback recursive search
    recursive_hits = sorted(
        glob.glob(os.path.join(data_root, "**", "*x3.png"), recursive=True)
    )
    if recursive_hits:
        return os.path.dirname(recursive_hits[0])

    raise FileNotFoundError(
        f"Cannot find a DIV2K LR x3 folder under data_root: {data_root}"
    )


def list_lr_pngs(lr_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(lr_dir, "*.png")))
    if not files:
        raise FileNotFoundError(f"No png found in: {lr_dir}")
    return files


def _extract_signature_input_name(sig_info, fallback_input_name: str) -> str:
    if isinstance(sig_info, dict):
        inputs = sig_info.get("inputs", None)
        if isinstance(inputs, (list, tuple)) and len(inputs) >= 1:
            return str(inputs[0])
        if isinstance(inputs, dict) and len(inputs) >= 1:
            return str(next(iter(inputs.keys())))
    return fallback_input_name


def load_tflite_input_spec(
    tflite_path: str,
) -> Tuple[str, str, Tuple[int, ...], np.dtype, Tuple[float, int]]:
    # Prefer ai_edge_litert, fall back to tensorflow.lite
    interpreter = None

    try:
        from ai_edge_litert.interpreter import Interpreter
        interpreter = Interpreter(model_path=tflite_path)
    except Exception:
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
        except Exception as e:
            raise RuntimeError(
                "Cannot create a TFLite interpreter. Install ai-edge-litert-nightly "
                "or tensorflow."
            ) from e

    interpreter.allocate_tensors()

    sig_list = interpreter.get_signature_list()
    if not sig_list:
        raise RuntimeError(
            "Model has no signature. ai_edge_quantizer.calibrate() expects calibration "
            "data keyed by signature name."
        )
    if len(sig_list) != 1:
        raise RuntimeError(
            f"Expected exactly 1 signature, got: {list(sig_list.keys())}"
        )

    signature_key = next(iter(sig_list.keys()))
    sig_info = sig_list[signature_key]

    detail = interpreter.get_input_details()[0]
    tensor_input_name = detail["name"]
    signature_input_name = _extract_signature_input_name(sig_info, tensor_input_name)

    shape = tuple(int(x) for x in detail["shape"])
    dtype = detail["dtype"]
    qparams = detail.get("quantization", (0.0, 0))
    return signature_key, signature_input_name, shape, dtype, qparams


def make_model_input(img_hwc_255: np.ndarray, input_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Supports:
      NCHW float/int
      NHWC float/int
    """
    if len(input_shape) != 4:
        raise RuntimeError(f"Expected 4D input tensor, got shape={input_shape}")

    n, a, b, c = input_shape

    if n != 1:
        raise RuntimeError(f"Expected batch=1 for calibration, got shape={input_shape}")

    # Heuristic:
    # NCHW if second dim in {1,3}
    # NHWC if last dim in {1,3}
    if a in (1, 3):  # NCHW
        out_h, out_w = b, c
        x = resize_hw(img_hwc_255, out_h, out_w)
        x = np.transpose(x, (2, 0, 1))[None, ...]  # 1CHW
        return x.astype(np.float32)

    if c in (1, 3):  # NHWC
        out_h, out_w = a, b
        x = resize_hw(img_hwc_255, out_h, out_w)
        x = x[None, ...]  # 1HWC
        return x.astype(np.float32)

    raise RuntimeError(f"Cannot infer input layout from shape={input_shape}")


def calibration_dict_generator(
    image_paths: Iterable[str],
    input_name: str,
    input_shape: Tuple[int, ...],
    num_samples: int,
) -> Generator[dict, None, None]:
    cnt = 0
    for p in image_paths:
        img = read_rgb(p)
        x = make_model_input(img, input_shape)
        yield {input_name: x}
        cnt += 1
        if cnt >= num_samples:
            break


def pick_recipe(recipe_mod, recipe_name: str):
    if hasattr(recipe_mod, recipe_name):
        return getattr(recipe_mod, recipe_name)()

    available = [x for x in dir(recipe_mod) if not x.startswith("_")]
    raise RuntimeError(
        f"Recipe '{recipe_name}' not found in ai_edge_quantizer.recipe. "
        f"Available: {available}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--float_tflite", type=str, required=True)
    ap.add_argument("--out_tflite", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--num_calib", type=int, default=64)
    ap.add_argument(
        "--recipe",
        type=str,
        default="static_wi8_ai8",
        help="Recommended for NPU: static_wi8_ai8",
    )
    args = ap.parse_args()

    from ai_edge_quantizer import quantizer, recipe

    signature_key, input_name, input_shape, input_dtype, _ = load_tflite_input_spec(
        args.float_tflite
    )
    print(f"[info] signature_key={signature_key}")
    print(f"[info] input_name={input_name}")
    print(f"[info] input_shape={input_shape}")
    print(f"[info] input_dtype={input_dtype}")

    lr_dir = detect_div2k_lr_dir(args.data_root)
    print(f"[info] calibration source dir={lr_dir}")

    image_paths = list_lr_pngs(lr_dir)
    print(f"[info] num_candidate_images={len(image_paths)}")

    qt = quantizer.Quantizer(args.float_tflite)
    qt.load_quantization_recipe(pick_recipe(recipe, args.recipe))

    if not hasattr(qt, "calibrate"):
        raise RuntimeError(
            "This ai_edge_quantizer build does not expose Quantizer.calibrate(). "
            "Static INT8 quantization needs calibration."
        )

    calibration_data = {
        signature_key: calibration_dict_generator(
            image_paths=image_paths,
            input_name=input_name,
            input_shape=input_shape,
            num_samples=args.num_calib,
        )
    }

    cal_result = qt.calibrate(calibration_data)
    print("[OK] calibration succeeded.")

    out_dir = os.path.dirname(os.path.abspath(args.out_tflite))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        qt.quantize(cal_result).export_model(args.out_tflite)
    except TypeError:
        # Some versions may keep calibration state inside the quantizer
        qt.quantize().export_model(args.out_tflite)

    print(f"[OK] wrote quantized TFLite: {args.out_tflite}")


if __name__ == "__main__":
    main()