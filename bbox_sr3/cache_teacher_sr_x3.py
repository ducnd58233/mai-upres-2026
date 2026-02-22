import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

from bbox_sr3.mambair_teacher import load_teacher


def pil_to_tensor_255(p: str) -> torch.Tensor:
    img = Image.open(p).convert("RGB")
    arr = np.array(img, dtype=np.float32)  # HWC 0..255
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1CHW
    return t


def save_png_u8(path: str, sr255: torch.Tensor):
    # sr255: 1,3,H,W float
    arr = sr255.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_npy_f16(path: str, sr255: torch.Tensor):
    # WARNING: big files
    arr = sr255.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float16)
    np.save(path, arr)


def save_npz_f16(path: str, sr255: torch.Tensor):
    # compressed, still heavier than png for natural images
    arr = sr255.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float16)
    np.savez_compressed(path, arr=arr)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mambair_repo", type=str, required=True)
    ap.add_argument("--teacher_ckpt", type=str, required=True)
    ap.add_argument("--train_lr_dir", type=str, required=True)
    ap.add_argument("--ids_txt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="teacher_cache_x3")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fmt", type=str, default="png", choices=["png", "npy", "npz"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.ids_txt, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    ids = sorted(set(ids))

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    teacher = load_teacher(args.mambair_repo, args.teacher_ckpt, device=str(device))

    for stem in tqdm(ids, desc="Cache teacher SR"):
        lr_path = os.path.join(args.train_lr_dir, f"{stem}x{args.scale}.png")
        if not os.path.exists(lr_path):
            continue

        lr255 = pil_to_tensor_255(lr_path).to(device)  # 1,3,h,w in 0..255
        _, _, h_lr, w_lr = lr255.shape

        with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
            sr01 = teacher(lr255 / 255.0)  # 0..1
        sr255 = torch.clamp(sr01 * 255.0, 0.0, 255.0)

        # Crop to exact expected size (avoid off-by-one from padding)
        exp_h, exp_w = h_lr * args.scale, w_lr * args.scale
        sr255 = sr255[..., :exp_h, :exp_w]

        if args.fmt == "png":
            out_path = os.path.join(args.out_dir, f"{stem}.png")
            save_png_u8(out_path, sr255)
        elif args.fmt == "npz":
            out_path = os.path.join(args.out_dir, f"{stem}.npz")
            save_npz_f16(out_path, sr255)
        else:
            out_path = os.path.join(args.out_dir, f"{stem}.npy")
            save_npy_f16(out_path, sr255)

    print("[OK] saved cache to:", args.out_dir)


if __name__ == "__main__":
    main()