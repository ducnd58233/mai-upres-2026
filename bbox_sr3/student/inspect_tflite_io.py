import argparse
from typing import Dict, Any, Tuple

import numpy as np


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


def tensor_summary(detail: Dict[str, Any]) -> Dict[str, Any]:
    shape = tuple(int(x) for x in detail["shape"])
    shape_sig = tuple(int(x) for x in detail.get("shape_signature", shape))
    q = detail.get("quantization", (0.0, 0))
    qp = detail.get("quantization_parameters", {})

    return {
        "name": detail.get("name", ""),
        "index": detail.get("index", -1),
        "shape": shape,
        "shape_signature": shape_sig,
        "dtype": str(detail.get("dtype", "")),
        "layout_guess": infer_layout(shape),
        "quantization": q,
        "scales": qp.get("scales", None),
        "zero_points": qp.get("zero_points", None),
        "quantized_dimension": qp.get("quantized_dimension", None),
    }


def print_tensor_info(title: str, detail: Dict[str, Any]) -> None:
    s = tensor_summary(detail)
    print(f"\n[{title}]")
    print(f"name              : {s['name']}")
    print(f"index             : {s['index']}")
    print(f"shape             : {s['shape']}")
    print(f"shape_signature   : {s['shape_signature']}")
    print(f"dtype             : {s['dtype']}")
    print(f"layout_guess      : {s['layout_guess']}")
    print(f"quantization      : {s['quantization']}")

    scales = s["scales"]
    zero_points = s["zero_points"]
    qdim = s["quantized_dimension"]

    if scales is not None and len(scales) > 0:
        if len(scales) <= 8:
            print(f"q_scales          : {np.asarray(scales)}")
            print(f"q_zero_points     : {np.asarray(zero_points)}")
        else:
            print(f"q_scales          : len={len(scales)}")
            print(f"q_zero_points     : len={len(zero_points)}")
        print(f"q_dim             : {qdim}")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--input_h", type=int, default=None)
    ap.add_argument("--input_w", type=int, default=None)
    ap.add_argument("--show_all_tensors", action="store_true")
    args = ap.parse_args()

    itp, backend = load_interpreter(args.model, num_threads=args.threads)
    print(f"[backend] {backend}")

    input_details = itp.get_input_details()
    output_details = itp.get_output_details()

    print(f"num_inputs         : {len(input_details)}")
    print(f"num_outputs        : {len(output_details)}")

    input_detail = input_details[0]
    input_detail = maybe_resize_input_tensor(
        itp, input_detail, args.input_h, args.input_w
    )
    output_detail = itp.get_output_details()[0]

    print_tensor_info("INPUT", input_detail)
    print_tensor_info("OUTPUT", output_detail)

    if args.show_all_tensors:
        print("\n[ALL TENSORS]")
        for d in itp.get_tensor_details():
            s = tensor_summary(d)
            print(
                f"idx={s['index']:4d} "
                f"shape={s['shape']} "
                f"dtype={s['dtype']} "
                f"layout={s['layout_guess']} "
                f"name={s['name']}"
            )


if __name__ == "__main__":
    main()