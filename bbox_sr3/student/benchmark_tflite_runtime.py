import argparse
import time
import math
from typing import Dict, Any, Tuple, List

import numpy as np
from PIL import Image


def load_interpreter(model_path: str, num_threads: int = 1):
    try:
        from ai_edge_litert.interpreter import Interpreter
        itp = Interpreter(model_path=model_path, num_threads=num_threads)
        backend = "ai_edge_litert"
    except Exception:
        import tensorflow as tf
        itp = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        backend = "tensorflow.lite"

    itp.allocate_tensors()
    return itp, backend


def infer_layout(shape: Tuple[int, ...]) -> str:
    if len(shape) != 4:
        return "not_4d"

    _, d1, d2, d3 = shape
    is_nchw = d1 in (1, 3, 4)
    is_nhwc = d3 in (1, 3, 4)

    if is_nchw and not is_nhwc:
        return "NCHW"
    if is_nhwc and not is_nchw:
        return "NHWC"
    if is_nchw and is_nhwc:
        return "ambiguous"
    return "unknown"


def maybe_resize_input_tensor(itp, input_detail: Dict[str, Any], input_h: int, input_w: int):
    shape = list(int(x) for x in input_detail["shape"])
    shape_sig = list(int(x) for x in input_detail.get("shape_signature", shape))

    if len(shape) != 4:
        return input_detail

    has_dynamic = any(v <= 0 for v in shape_sig)
    if not has_dynamic:
        return input_detail

    if input_h is None or input_w is None:
        raise RuntimeError(
            "Model có dynamic input shape. Hãy truyền thêm --input_h và --input_w."
        )

    layout = infer_layout(tuple(shape if all(v > 0 for v in shape) else shape_sig))
    if layout == "NCHW":
        c = shape[1] if shape[1] > 0 else 3
        new_shape = [1, c, input_h, input_w]
    elif layout == "NHWC":
        c = shape[3] if shape[3] > 0 else 3
        new_shape = [1, input_h, input_w, c]
    else:
        raise RuntimeError(
            f"Không đoán được layout cho dynamic shape: shape={shape}, shape_signature={shape_sig}"
        )

    itp.resize_tensor_input(input_detail["index"], new_shape, strict=False)
    itp.allocate_tensors()
    return itp.get_input_details()[0]


def load_rgb_image(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)


def resize_hwc(img_hwc: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    pil = Image.fromarray(np.clip(img_hwc, 0, 255).astype(np.uint8))
    pil = pil.resize((out_w, out_h), Image.BICUBIC)
    return np.array(pil, dtype=np.float32)


def make_input_from_image(img_hwc_255: np.ndarray, input_shape: Tuple[int, ...]) -> np.ndarray:
    if len(input_shape) != 4 or input_shape[0] != 1:
        raise RuntimeError(f"Unsupported input shape: {input_shape}")

    layout = infer_layout(input_shape)

    if layout == "NCHW":
        c, h, w = input_shape[1], input_shape[2], input_shape[3]
        x = resize_hwc(img_hwc_255, h, w)
        if c == 1:
            x = np.mean(x, axis=2, keepdims=True)
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return x.astype(np.float32)

    if layout == "NHWC":
        h, w, c = input_shape[1], input_shape[2], input_shape[3]
        x = resize_hwc(img_hwc_255, h, w)
        if c == 1:
            x = np.mean(x, axis=2, keepdims=True)
        x = x[None, ...]
        return x.astype(np.float32)

    raise RuntimeError(f"Không đoán được input layout từ shape={input_shape}")


def make_random_input(input_shape: Tuple[int, ...], seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 255.0, size=input_shape).astype(np.float32)


def quantize_input_if_needed(x_float: np.ndarray, input_detail: Dict[str, Any]) -> np.ndarray:
    dtype = input_detail["dtype"]
    scale, zero_point = input_detail.get("quantization", (0.0, 0))

    if np.issubdtype(dtype, np.integer):
        if scale <= 0:
            raise RuntimeError(f"Invalid input quantization params: {input_detail.get('quantization')}")
        q = np.round(x_float / scale + zero_point)
        q = np.clip(q, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
        return q

    return x_float.astype(dtype)


def benchmark_invoke(itp, input_detail: Dict[str, Any], x: np.ndarray, warmup: int, repeat: int) -> List[float]:
    times_ms: List[float] = []

    for _ in range(max(0, warmup)):
        itp.set_tensor(input_detail["index"], x)
        itp.invoke()

    for _ in range(max(1, repeat)):
        itp.set_tensor(input_detail["index"], x)
        t0 = time.perf_counter()
        itp.invoke()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return times_ms


def print_stats(times_ms: List[float]) -> None:
    arr = np.asarray(times_ms, dtype=np.float64)
    print("\n[runtime]")
    print(f"runs              : {len(arr)}")
    print(f"min_ms            : {arr.min():.4f}")
    print(f"mean_ms           : {arr.mean():.4f}")
    print(f"median_ms         : {np.median(arr):.4f}")
    print(f"p90_ms            : {np.percentile(arr, 90):.4f}")
    print(f"p95_ms            : {np.percentile(arr, 95):.4f}")
    print(f"max_ms            : {arr.max():.4f}")
    fps = 1000.0 / arr.mean() if arr.mean() > 0 else math.inf
    print(f"approx_fps        : {fps:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--image", type=str, default="")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--input_h", type=int, default=None)
    ap.add_argument("--input_w", type=int, default=None)
    args = ap.parse_args()

    itp, backend = load_interpreter(args.model, num_threads=args.threads)
    print(f"[backend] {backend}")

    input_detail = itp.get_input_details()[0]
    input_detail = maybe_resize_input_tensor(
        itp, input_detail, args.input_h, args.input_w
    )

    input_shape = tuple(int(x) for x in input_detail["shape"])
    layout = infer_layout(input_shape)

    print(f"input_shape        : {input_shape}")
    print(f"layout_guess       : {layout}")
    print(f"input_dtype        : {input_detail['dtype']}")
    print(f"input_quant        : {input_detail.get('quantization', (0.0, 0))}")

    if args.image:
        img = load_rgb_image(args.image)
        x_float = make_input_from_image(img, input_shape)
        print(f"input_source       : image ({args.image})")
    else:
        x_float = make_random_input(input_shape, seed=args.seed)
        print("input_source       : random")

    x = quantize_input_if_needed(x_float, input_detail)
    print(f"actual_input_dtype : {x.dtype}")
    print(f"actual_input_min   : {x.min()}")
    print(f"actual_input_max   : {x.max()}")

    times_ms = benchmark_invoke(
        itp=itp,
        input_detail=input_detail,
        x=x,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    print_stats(times_ms)


if __name__ == "__main__":
    main()