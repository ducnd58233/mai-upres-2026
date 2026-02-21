#!/usr/bin/env python3
import argparse
import glob
import random
import time
import numpy as np
from PIL import Image

from ai_edge_quantizer import quantizer
from ai_edge_quantizer import recipe
from ai_edge_quantizer.utils import tfl_interpreter_utils

# =============================================================================
# PATCH AEQ calibration interpreter:
# - Force allocate_tensors
# - Optionally preserve_all_tensors
# - Disable XNNPACK to keep tensors readable (stability)
# - Skip unreadable tensors to avoid crash
# =============================================================================

_ORIG_CREATE = tfl_interpreter_utils.create_tfl_interpreter
_ORIG_GET_MAP = tfl_interpreter_utils.get_tensor_name_to_content_map

# these globals will be set from args in main()
_PATCH_PRESERVE_ALL_TENSORS = False
_PATCH_DISABLE_XNNPACK = True

def _create_tfl_interpreter_for_calib(tflite_model, *args, **kwargs):
    kwargs["allocate_tensors"] = True
    kwargs["preserve_all_tensors"] = bool(_PATCH_PRESERVE_ALL_TENSORS)
    kwargs["use_xnnpack"] = (not _PATCH_DISABLE_XNNPACK)
    return _ORIG_CREATE(tflite_model, *args, **kwargs)

def _safe_get_tensor_name_to_content_map(tflite_interpreter, subgraph_index=0, dequantize=False):
    tensors = {}
    for td in tflite_interpreter.get_tensor_details(subgraph_index):
        name = td.get("name", "")
        if not name or "scratch" in name:
            continue
        if "shape" in td and (not np.all(td["shape"])):
            continue
        try:
            tensors[name] = tfl_interpreter_utils.get_tensor_data(
                tflite_interpreter, td, subgraph_index, dequantize
            )
        except Exception as e:
            # Most common: ValueError("Tensor data is null. Run allocate_tensors() first")
            msg = str(e).lower()
            if "tensor data is null" in msg:
                continue
            # keep skipping other unreadable cases quietly
            if "scratch" in name:
                continue
            # if it's a different error, raise to avoid hiding real bugs
            raise
    return tensors

# apply patches
tfl_interpreter_utils.create_tfl_interpreter = _create_tfl_interpreter_for_calib
tfl_interpreter_utils.get_tensor_name_to_content_map = _safe_get_tensor_name_to_content_map


def load_rgb_255(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32)  # HWC, [0..255]


def crop_or_resize(arr_hwc: np.ndarray, H: int, W: int) -> np.ndarray:
    h, w = arr_hwc.shape[:2]
    if h < H or w < W:
        img = Image.fromarray(np.clip(arr_hwc, 0, 255).astype(np.uint8))
        img = img.resize((W, H), resample=Image.BICUBIC)
        return np.array(img, dtype=np.float32)

    y = 0 if h == H else random.randint(0, h - H)
    x = 0 if w == W else random.randint(0, w - W)
    return arr_hwc[y:y + H, x:x + W, :]


def make_input_blob(arr_hwc: np.ndarray, in_shape: np.ndarray) -> np.ndarray:
    shape = tuple(int(x) for x in in_shape)
    if len(shape) != 4:
        raise ValueError(f"Unexpected input shape: {shape}")

    # NHWC
    if shape[-1] == 3:
        _, H, W, C = shape
        assert C == 3
        patch = crop_or_resize(arr_hwc, H, W)
        return patch[None, ...].astype(np.float32)

    # NCHW
    if shape[1] == 3:
        _, C, H, W = shape
        assert C == 3
        patch = crop_or_resize(arr_hwc, H, W)
        patch = np.transpose(patch, (2, 0, 1))[None, ...]
        return patch.astype(np.float32)

    raise ValueError(f"Cannot infer layout from shape: {shape}")


def main():
    global _PATCH_PRESERVE_ALL_TENSORS, _PATCH_DISABLE_XNNPACK

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tflite", type=str, required=True)
    ap.add_argument("--out_tflite", type=str, required=True)
    ap.add_argument("--lr_dir", type=str, required=True, help="DIV2K_train_LR_bicubic/X3 (png)")
    ap.add_argument("--num_calib_images", type=int, default=800)
    ap.add_argument("--crops_per_image", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)

    # debug/progress controls
    ap.add_argument("--progress_every", type=int, default=50, help="print every N samples")
    ap.add_argument("--max_steps", type=int, default=0, help="0=use all; else limit total yielded samples")
    ap.add_argument("--preserve_all_tensors", action="store_true", help="more stable, but uses more RAM")
    ap.add_argument("--enable_xnnpack", action="store_true", help="faster but may trigger unreadable tensors")
    args = ap.parse_args()

    _PATCH_PRESERVE_ALL_TENSORS = bool(args.preserve_all_tensors)
    _PATCH_DISABLE_XNNPACK = (not args.enable_xnnpack)

    random.seed(args.seed)
    np.random.seed(args.seed)

    # create interpreter just to inspect signatures/input
    itp = tfl_interpreter_utils.create_tfl_interpreter(args.in_tflite, num_threads=args.threads)
    if hasattr(itp, "allocate_tensors"):
        try:
            itp.allocate_tensors()
        except Exception:
            pass

    sigs = itp.get_signature_list()
    if not sigs:
        raise RuntimeError(
            "Model has no signatures. AI Edge Quantizer calibrates via signature runner; "
            "hãy export lại bằng litert_torch để có 'serving_default'."
        )
    print("Signatures:", list(sigs.keys()))

    lr_paths = sorted(glob.glob(args.lr_dir.rstrip("/") + "/*.png"))
    if not lr_paths:
        raise FileNotFoundError(f"No png found in {args.lr_dir}")

    random.shuffle(lr_paths)
    lr_paths = lr_paths[: max(args.num_calib_images, 1)]

    calib_data = {}

    for sig_key in sigs.keys():
        runner = itp.get_signature_runner(sig_key)
        in_details = runner.get_input_details()
        if len(in_details) != 1:
            raise RuntimeError(f"Expected single input, got {list(in_details.keys())} for signature={sig_key}")

        in_name = list(in_details.keys())[0]
        in_shape = in_details[in_name]["shape"]
        total = len(lr_paths) * max(1, args.crops_per_image)
        if args.max_steps and args.max_steps > 0:
            total = min(total, args.max_steps)

        print(f"[{sig_key}] input name={in_name}, shape={tuple(in_shape)} dtype={in_details[in_name]['dtype']}")
        print(f"[{sig_key}] calib samples planned: {total} "
              f"(imgs={len(lr_paths)} crops/img={max(1, args.crops_per_image)} max_steps={args.max_steps})")
        print(f"[{sig_key}] calib opts: preserve_all_tensors={args.preserve_all_tensors} "
              f"xnnpack={'ON' if args.enable_xnnpack else 'OFF'} threads={args.threads}")

        def gen(paths=lr_paths, in_shape=in_shape, in_name=in_name, total=total):
            t0 = time.time()
            step = 0
            for p in paths:
                arr = load_rgb_255(p)
                for _ in range(max(1, args.crops_per_image)):
                    step += 1
                    blob = make_input_blob(arr, in_shape)
                    if args.progress_every > 0 and (step == 1 or step % args.progress_every == 0):
                        dt = time.time() - t0
                        sps = step / dt if dt > 0 else 0.0
                        print(f"[calib {sig_key}] step {step}/{total}  ({sps:.2f} samples/s)", flush=True)
                    yield {in_name: blob}
                    if total and step >= total:
                        return

        calib_data[sig_key] = gen()

    q = quantizer.Quantizer(args.in_tflite)
    q.load_quantization_recipe(recipe.static_wi8_ai8())

    print("==> Start calibration ...", flush=True)
    calib_res = q.calibrate(calib_data, num_threads=args.threads)

    print("==> Quantizing ...", flush=True)
    result = q.quantize(calib_res)
    result.export_model(args.out_tflite, overwrite=True)

    print("Saved INT8:", args.out_tflite)


if __name__ == "__main__":
    main()