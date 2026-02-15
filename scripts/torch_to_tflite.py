import argparse
import logging
import os
from pathlib import Path

import tensorflow as tf
import torch

from configs import configure_logging
from training import get_registered_model_types, get_required_args, load_for_export

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
configure_logging()
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_OUTPUT_DIR = _PROJECT_ROOT / "runs" / "models" / "export"


def to_onnx(model: torch.nn.Module, input_shape: list[int], out_path: str) -> None:
    """Export PyTorch model to ONNX."""
    logger.info("Exporting PyTorch -> ONNX  (%s)", out_path)
    size = tuple(int(d) for d in input_shape)
    dummy = torch.randn(size)

    torch.onnx.export(
        model,
        (dummy,),
        out_path,
        opset_version=18,
        do_constant_folding=True,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
    )


def to_tf(onnx_path: str, out_dir: str) -> None:
    """Convert ONNX to TF SavedModel."""
    import onnx2tf

    logger.info("Converting ONNX -> TF SavedModel  (%s/)", out_dir)
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=out_dir,
        non_verbose=True,
    )


def to_tflite(tf_dir: str, out_path: str) -> None:
    """Convert TF SavedModel to TFLite."""
    logger.info("Converting TF SavedModel -> TFLite  (%s)", out_path)
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
    tflite_bytes = converter.convert()
    if not isinstance(tflite_bytes, (bytes, bytearray)):
        raise TypeError("TFLite converter did not return bytes")
    Path(out_path).write_bytes(tflite_bytes)


def parse_args() -> argparse.Namespace:
    model_types = get_registered_model_types()
    p = argparse.ArgumentParser(
        description="Convert a PyTorch model to TFLite via ONNX. Use --model-type to select loader.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        help="Path to PyTorch checkpoint (.pt).",
    )
    p.add_argument(
        "--model-type",
        choices=model_types,
        default="raw",
        help="Model type (registered loader). Default: raw.",
    )
    p.add_argument(
        "--height",
        type=int,
        default=None,
        help="Input height (required for some model types, e.g. antsr).",
    )
    p.add_argument(
        "--width",
        type=int,
        default=None,
        help="Input width (required for some model types, e.g. antsr).",
    )
    p.add_argument(
        "--input-shape",
        nargs="+",
        type=int,
        metavar="DIM",
        help="Input tensor shape, e.g. 1 3 224 224 (required for model-type raw).",
    )
    args = p.parse_args()

    required = get_required_args(args.model_type)
    missing = [r for r in required if getattr(args, r.replace("-", "_"), None) is None]
    if missing:
        # Normalize: required_args use underscores (height, width, input_shape)
        p.error(
            f"--model-type {args.model_type} requires: {required}. Missing: {missing}"
        )

    return args


def main() -> None:
    args = parse_args()

    out_dir = Path(EXPORT_OUTPUT_DIR)
    stem = Path(args.model).stem
    out_path = out_dir / f"{stem}.tflite"
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = str(out_dir / f"{stem}.onnx")
    tf_dir = str(out_dir / f"{stem}_tf")

    # Build kwargs for the loader from args
    kwargs: dict[str, object] = {}
    for key in get_required_args(args.model_type):
        val = getattr(args, key, None)
        if val is not None:
            kwargs[key] = val

    logger.info("Model      : %s", args.model)
    logger.info("Model type : %s", args.model_type)
    logger.info("Output     : %s", out_path)

    model, input_shape = load_for_export(args.model_type, args.model, **kwargs)
    logger.info("Input shape: %s", input_shape)

    to_onnx(model, input_shape, onnx_path)
    to_tf(onnx_path, tf_dir)
    to_tflite(tf_dir, str(out_path))

    logger.info("Done! TFLite saved to %s", out_path)


if __name__ == "__main__":
    main()
