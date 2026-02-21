import os, argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

from bbox_sr3.mambair_teacher import load_teacher

def pil_to_tensor_255(p):
    img = Image.open(p).convert("RGB")
    arr = np.array(img, dtype=np.float32)  # HWC 0..255
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mambair_repo", type=str, required=True)
    ap.add_argument("--teacher_ckpt", type=str, required=True)
    ap.add_argument("--train_lr_dir", type=str, required=True)
    ap.add_argument("--ids_txt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="teacher_cache_x3")
    ap.add_argument("--scale", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.ids_txt, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    ids = sorted(set(ids))

    teacher = load_teacher(args.mambair_repo, args.teacher_ckpt, device="cuda")

    for stem in tqdm(ids, desc="Cache teacher SR"):
        lr_path = os.path.join(args.train_lr_dir, f"{stem}x{args.scale}.png")
        if not os.path.exists(lr_path):
            continue
        lr255 = pil_to_tensor_255(lr_path).cuda()
        sr = teacher(lr255 / 255.0) * 255.0
        sr = torch.clamp(sr, 0.0, 255.0)
        arr = sr.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.float16)
        np.save(os.path.join(args.out_dir, f"{stem}.npy"), arr)

    print("[OK] saved cache to:", args.out_dir)

if __name__ == "__main__":
    main()