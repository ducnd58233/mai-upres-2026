import os
import glob
import argparse
import heapq
import numpy as np
from PIL import Image
from tqdm import tqdm


def pil_rgb(p: str) -> Image.Image:
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
def compute_target_embedding(model, preprocess, query_paths, W, H, aspect_crop, batch=64):
    import torch
    # accumulate mean embedding without storing all feats
    s = None
    n = 0

    for i in tqdm(range(0, len(query_paths), batch), desc="CLIP encode query"):
        ps = query_paths[i:i + batch]
        imgs = [to_domain_720p(pil_rgb(p), W, H, aspect_crop) for p in ps]
        x = torch.stack([preprocess(im) for im in imgs]).cuda()
        f = model.encode_image(x).float()
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-12)
        f_sum = f.sum(dim=0, keepdim=False)  # (D,)
        if s is None:
            s = f_sum
        else:
            s = s + f_sum
        n += f.shape[0]

    tgt = (s / max(1, n)).unsqueeze(0)  # (1,D)
    tgt = tgt / (tgt.norm(dim=-1, keepdim=True) + 1e-12)
    return tgt  # torch tensor on cuda


@torch.no_grad()
def stream_topk_candidates(model, preprocess, candidate_paths, tgt, k, W, H, aspect_crop, batch=64):
    import torch
    # heap keeps smallest sim at root
    heap = []  # list of (sim_float, stem)

    for i in tqdm(range(0, len(candidate_paths), batch), desc="CLIP encode candidates"):
        ps = candidate_paths[i:i + batch]
        imgs = [to_domain_720p(pil_rgb(p), W, H, aspect_crop) for p in ps]
        x = torch.stack([preprocess(im) for im in imgs]).cuda()
        f = model.encode_image(x).float()
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-12)

        sims = (f * tgt).sum(dim=-1).detach().cpu().numpy()  # (B,)
        for p, sim in zip(ps, sims):
            stem = os.path.splitext(os.path.basename(p))[0]
            item = (float(sim), stem)
            if len(heap) < k:
                heapq.heappush(heap, item)
            else:
                if item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)

    # sort desc
    heap.sort(key=lambda x: -x[0])
    return heap


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_hr_dir", type=str, required=True)
    ap.add_argument("--query_lr_dir", type=str, required=True, help="validation LR folder as query set")
    ap.add_argument("--k", type=int, default=2000)
    ap.add_argument("--out_txt", type=str, default="filtered_ids.txt")
    ap.add_argument("--clip_model", type=str, default="ViT-H-14")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--W", type=int, default=1280)
    ap.add_argument("--H", type=int, default=720)
    ap.add_argument("--aspect_crop", action="store_true", help="center-crop to 16:9 before resize")
    args = ap.parse_args()

    hr_paths = sorted(glob.glob(os.path.join(args.train_hr_dir, "*.png")))
    q_paths = sorted(glob.glob(os.path.join(args.query_lr_dir, "*.png")))
    if not hr_paths or not q_paths:
        raise FileNotFoundError("Empty train_hr_dir or query_lr_dir")

    model, preprocess = load_clip(args.clip_model)

    # Query: validation LR -> normalize to 720p domain
    tgt = compute_target_embedding(
        model, preprocess, q_paths,
        W=args.W, H=args.H, aspect_crop=args.aspect_crop,
        batch=args.batch
    )

    # Candidates: train HR -> normalize to 720p domain
    topk = stream_topk_candidates(
        model, preprocess, hr_paths, tgt,
        k=args.k, W=args.W, H=args.H, aspect_crop=args.aspect_crop,
        batch=args.batch
    )

    with open(args.out_txt, "w") as f:
        for sim, stem in topk:
            f.write(stem + "\n")

    print(f"[OK] Saved top-{args.k} ids -> {args.out_txt}")


if __name__ == "__main__":
    main()