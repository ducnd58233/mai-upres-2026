import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import logging
from pathlib import Path

import tensorflow as tf
import torch

DEFAULT_OUTPUT_DIR = "runs/models/export"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(model_path: str) -> torch.nn.Module:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    model = torch.load(str(path), map_location="cpu", weights_only=False)

    logger.info("Loaded PyTorch nn.Module checkpoint.")

    model.eval()
    for module in model.modules():
        module.training = False

    return model


def to_onnx(model: torch.nn.Module, input_shape: list, out_path: str) -> None:
    logger.info(f"[1/3] Exporting PyTorch -> ONNX  ({out_path})")
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
    import onnx2tf

    logger.info(f"[2/3] Converting ONNX -> TF SavedModel  ({out_dir}/)")
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=out_dir,
        non_verbose=True,
    )


def to_tflite(tf_dir: str, out_path: str) -> None:
    logger.info(f"[3/3] Converting TF SavedModel -> TFLite  ({out_path})")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)

    tflite_bytes = converter.convert()
    assert isinstance(tflite_bytes, (bytes, bytearray))
    Path(out_path).write_bytes(tflite_bytes)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a PyTorch model to TFLite via ONNX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        help=(
            "Path to a pickled PyTorch nn.Module file (.pt / .pth), "
            "saved via torch.save(model, path)"
        ),
    )
    p.add_argument(
        "--input-shape",
        nargs="+",
        type=int,
        required=True,
        metavar="DIM",
        help="Input tensor shape, e.g.  1 3 224 224",
    )
    p.add_argument(
        "--output",
        default=os.path.join(DEFAULT_OUTPUT_DIR, "model.tflite"),
        help=f"Output .tflite file path  (default: {DEFAULT_OUTPUT_DIR}/model.tflite)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_path = Path(args.output)
    out_dir = out_path.parent
    stem = out_path.stem

    os.makedirs(out_dir, exist_ok=True)

    onnx_path = str(out_dir / f"{stem}.onnx")
    tf_dir = str(out_dir / f"{stem}_tf")

    logger.info(f"Model      : {args.model}")
    logger.info(f"Input shape: {args.input_shape}")
    logger.info(f"Output     : {args.output}")

    model = load_model(args.model)
    to_onnx(model, args.input_shape, onnx_path)
    to_tf(onnx_path, tf_dir)
    to_tflite(tf_dir, str(out_path))

    logger.info(f"Done! TFLite model saved to: {out_path}")


if __name__ == "__main__":
    main()
