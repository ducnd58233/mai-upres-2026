import os
import json
import argparse
from typing import Tuple

import numpy as np
import torch

from model_antsr import AntSR


def load_deploy_ckpt(ckpt_path: str, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt or "cfg" not in ckpt:
        raise RuntimeError(
            f"Invalid deploy checkpoint: {ckpt_path}. "
            "Expected keys: {'model', 'cfg'}"
        )

    cfg = dict(ckpt["cfg"])
    model = AntSR(deploy=True, **cfg).eval().to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    for p in model.parameters():
        p.requires_grad_(False)

    return model, cfg


@torch.inference_mode()
def run_torch(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    y = model(x)
    return y.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to ckpt_best_s3_qat_deploy.pt")
    ap.add_argument("--out_tflite", type=str, required=True,
                    help="Output float .tflite path")
    ap.add_argument("--meta_json", type=str, default="",
                    help="Optional metadata json output")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--input_h", type=int, default=720)
    ap.add_argument("--input_w", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-4)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    model, cfg = load_deploy_ckpt(args.ckpt, device=device)

    # AntSR in your repo expects float32 NCHW in [0..255]
    sample = torch.rand(1, 3, args.input_h, args.input_w, device=device) * 255.0

    torch_out = run_torch(model, sample)

    import litert_torch  # official converter path

    edge_model = litert_torch.convert(model.eval(), (sample,))

    # Optional local inference on LiteRT model
    edge_out = edge_model(sample)

    # Normalize edge output to numpy
    if isinstance(edge_out, (list, tuple)):
        if len(edge_out) != 1:
            raise RuntimeError(f"Unexpected number of outputs from LiteRT model: {len(edge_out)}")
        edge_out = edge_out[0]

    edge_out = np.asarray(edge_out)

    ok = np.allclose(torch_out, edge_out, atol=args.atol, rtol=args.rtol)
    max_abs = float(np.max(np.abs(torch_out - edge_out)))
    mean_abs = float(np.mean(np.abs(torch_out - edge_out)))

    out_dir = os.path.dirname(os.path.abspath(args.out_tflite))
    os.makedirs(out_dir, exist_ok=True)
    edge_model.export(args.out_tflite)

    print(f"[OK] exported float TFLite: {args.out_tflite}")
    print(f"[check] allclose={ok} max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}")

    if args.meta_json:
        meta = {
            "ckpt": os.path.abspath(args.ckpt),
            "out_tflite": os.path.abspath(args.out_tflite),
            "cfg": cfg,
            "input_shape": [1, 3, args.input_h, args.input_w],
            "output_shape": list(torch_out.shape),
            "torch_vs_litert_allclose": bool(ok),
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
        }
        with open(args.meta_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[OK] wrote metadata: {args.meta_json}")


if __name__ == "__main__":
    main()