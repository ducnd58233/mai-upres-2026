import os, glob, argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

def pil_rgb(p):
    return Image.open(p).convert("RGB")

def aspect_center_crop(img: Image.Image, target_ratio: float):
    w, h = img.size
    cur = w / h
    if abs(cur - target_ratio) < 1e-6:
        return img
    if cur > target_ratio:
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        return img.crop((x0, 0, x0 + new_w, h))
    else:
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        return img.crop((0, y0, w, y0 + new_h))

def to_domain_720p(img: Image.Image, W=1280, H=720, aspect_crop=True):
    if aspect_crop:
        img = aspect_center_crop(img, target_ratio=W / H)
    return img.resize((W, H), resample=Image.BICUBIC)

def load_clip(model_name="ViT-H-14"):
    import torch
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="laion2b_s32b_b79k"
    )
    model.eval().cuda()
    return model, preprocess

@torch.no_grad()
def encode_images(model, preprocess, pil_images, batch=64):
    import torch
    feats = []
    for i in tqdm(range(0, len(pil_images), batch), desc="CLIP encode"):
        imgs = pil_images[i:i+batch]
        x = torch.stack([preprocess(im) for im in imgs]).cuda()
        f = model.encode_image(x).float()
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-12)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)

def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_hr_dir", type=str, required=True)
    ap.add_argument("--query_lr_dir", type=str, required=True,
                    help="validation LR folder as query set")
    ap.add_argument("--k", type=int, default=2000)
    ap.add_argument("--out_txt", type=str, default="filtered_ids.txt")
    ap.add_argument("--clip_model", type=str, default="ViT-H-14")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--W", type=int, default=1280)
    ap.add_argument("--H", type=int, default=720)
    ap.add_argument("--aspect_crop", action="store_true",
                    help="center-crop to 16:9 before resize")
    args = ap.parse_args()

    hr_paths = sorted(glob.glob(os.path.join(args.train_hr_dir, "*.png")))
    q_paths  = sorted(glob.glob(os.path.join(args.query_lr_dir, "*.png")))
    if not hr_paths or not q_paths:
        raise FileNotFoundError("Empty train_hr_dir or query_lr_dir")

    model, preprocess = load_clip(args.clip_model)

    # Query: validation LR -> normalize to 720p domain
    q_imgs = [to_domain_720p(pil_rgb(p), args.W, args.H, args.aspect_crop) for p in q_paths]
    q_feat = encode_images(model, preprocess, q_imgs, batch=args.batch)
    tgt = q_feat.mean(axis=0, keepdims=True)
    tgt = tgt / (np.linalg.norm(tgt, axis=-1, keepdims=True) + 1e-12)

    # Candidates: train HR -> normalize to 720p domain (BBox-style)
    c_imgs = [to_domain_720p(pil_rgb(p), args.W, args.H, args.aspect_crop) for p in hr_paths]
    c_feat = encode_images(model, preprocess, c_imgs, batch=args.batch)
    c_feat = c_feat / (np.linalg.norm(c_feat, axis=-1, keepdims=True) + 1e-12)

    sim = (c_feat * tgt).sum(axis=-1)
    topk = np.argsort(-sim)[:args.k]

    with open(args.out_txt, "w") as f:
        for idx in topk:
            stem = os.path.splitext(os.path.basename(hr_paths[idx]))[0]
            f.write(stem + "\n")

    print(f"[OK] Saved top-{args.k} ids -> {args.out_txt}")

if __name__ == "__main__":
    main()