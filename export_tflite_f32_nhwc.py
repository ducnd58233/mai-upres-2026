#!/usr/bin/env python3
import argparse
import torch
import litert_torch
from model_antsr import AntSR

def is_deploy_state_dict(sd: dict) -> bool:
    # deploy ckpt thường có ".reparam." hoặc "reparam.weight"
    for k in sd.keys():
        if ".reparam." in k or k.endswith("reparam.weight") or k.endswith("reparam.bias"):
            return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="ckpt_best_s2_deploy.pt or ckpt_best_s3_qat_deploy.pt")
    ap.add_argument("--out", required=True, help="output .tflite")
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--trace_random", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    torch.set_grad_enabled(False)

    blob = torch.load(args.ckpt, map_location="cpu")
    if not (isinstance(blob, dict) and "cfg" in blob and "model" in blob):
        raise RuntimeError("Expected deploy/export ckpt with keys {'cfg','model'}")

    cfg = blob["cfg"]
    sd = blob["model"]

    deploy_sd = is_deploy_state_dict(sd)

    # IMPORTANT: build model with correct deploy flag and strict=True
    model = AntSR(deploy=deploy_sd, **cfg).eval()
    model.load_state_dict(sd, strict=True)

    # In case someone passes a non-deploy sd
    if not deploy_sd:
        model.switch_to_deploy()
        model.eval()

    # NHWC I/O
    model_nhwc = litert_torch.to_channel_last_io(model, args=[0]).eval()

    if args.trace_random:
        example = torch.rand((1, args.h, args.w, 3), dtype=torch.float32) * 255.0
    else:
        example = torch.zeros((1, args.h, args.w, 3), dtype=torch.float32)

    edge_model = litert_torch.convert(model_nhwc, (example,))
    edge_model.export(args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
