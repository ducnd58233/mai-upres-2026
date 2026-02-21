import argparse
import torch

from model_antsr import AntSR

# LiteRT Torch
import litert_torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="ckpt_best_s3_qat_deploy.pt (or s2_deploy)")
    ap.add_argument("--out", type=str, required=True, help="output .tflite")
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--w", type=int, default=1280)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["cfg"]
    cfg["out_clamp_mode"] = "none"

    sd  = ckpt["model"]

    # deploy=True để RepConv đã fuse (runtime-friendly)
    model = AntSR(deploy=True, **cfg).eval()
    model.load_state_dict(sd, strict=True)

    # IMPORTANT: export đúng size runtime eval
    example = torch.randn(1, 3, args.h, args.w)

    # Convert + export
    edge_model = litert_torch.convert(model, (example,))

    edge_model.export(args.out)

    print("Exported:", args.out)

if __name__ == "__main__":
    main()
