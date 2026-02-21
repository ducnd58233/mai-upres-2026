#!/usr/bin/env python3
import argparse, random
import numpy as np
from PIL import Image
import torch
import tflite_runtime.interpreter as tflite

from model_antsr import AntSR

def load_rgb_255(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)

def crop_or_resize(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if h < H or w < W:
        img = Image.fromarray(np.clip(arr,0,255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
        return np.array(img, np.float32)
    y = 0 if h == H else random.randint(0, h - H)
    x = 0 if w == W else random.randint(0, w - W)
    return arr[y:y+H, x:x+W, :].astype(np.float32)

def quantize_input(x: np.ndarray, dtype, qinfo):
    scale, zp = qinfo
    if scale == 0:
        return x.astype(dtype)
    q = np.round(x / scale + zp)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--pt_ckpt", required=True, help="deploy ckpt with keys {'cfg','model'}")
    ap.add_argument("--img", required=True, help="any LR png (HWC)")
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_grad_enabled(False)

    lr = load_rgb_255(args.img)
    lr = crop_or_resize(lr, args.h, args.w)  # HWC
    inp = lr[None, ...]  # 1,H,W,3

    # --- PyTorch ---
    blob = torch.load(args.pt_ckpt, map_location="cpu")
    cfg = blob["cfg"]
    sd = blob["model"]
    model = AntSR(deploy=True, **cfg).eval()
    model.load_state_dict(sd, strict=True)

    # model expects NCHW in your PyTorch forward
    lr_t = torch.from_numpy(np.transpose(lr, (2,0,1))[None, ...]).float()
    sr_pt = model(lr_t).detach().cpu().numpy()
    sr_pt = np.transpose(sr_pt[0], (1,2,0))  # HWC

    # --- TFLite ---
    itp = tflite.Interpreter(model_path=args.tflite, num_threads=4)
    itp.allocate_tensors()
    in0 = itp.get_input_details()[0]
    out0 = itp.get_output_details()[0]
    in_q = in0.get("quantization", (0.0, 0))
    out_q = out0.get("quantization", (0.0, 0))

    x = inp
    if in0["dtype"] != np.float32:
        x = quantize_input(x, in0["dtype"], in_q)
    else:
        x = x.astype(np.float32)

    itp.set_tensor(in0["index"], x)
    itp.invoke()
    y = itp.get_tensor(out0["index"])
    if out0["dtype"] != np.float32:
        sr_tf = dequant_output(y, out_q)[0]
    else:
        sr_tf = y.astype(np.float32)[0]

    # --- Compare ---
    sr_pt = np.clip(sr_pt, 0, 255).astype(np.float32)
    sr_tf = np.clip(sr_tf, 0, 255).astype(np.float32)

    H = min(sr_pt.shape[0], sr_tf.shape[0])
    W = min(sr_pt.shape[1], sr_tf.shape[1])
    sr_pt = sr_pt[:H,:W,:]
    sr_tf = sr_tf[:H,:W,:]

    diff = sr_tf - sr_pt
    print("Compare TFLite vs PyTorch")
    print("  max abs:", float(np.max(np.abs(diff))))
    print("  mean abs:", float(np.mean(np.abs(diff))))
    print("  mean:", float(np.mean(diff)))
    print("  std :", float(np.std(diff)))

if __name__ == "__main__":
    main()
