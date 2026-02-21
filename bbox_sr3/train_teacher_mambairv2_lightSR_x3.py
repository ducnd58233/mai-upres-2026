import os, math, argparse
from itertools import cycle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from bbox_sr3.mambair_teacher import build_mambairv2_lightsr_x3, _pick_state_dict
from bbox_sr3.teacher_dataset_mix import RandomHRMixOnTheFlyLR

def charbonnier(x, y, eps=1e-3):
    return torch.mean(torch.sqrt((x - y) ** 2 + eps * eps))

def psnr01(pred01: torch.Tensor, gt01: torch.Tensor, eps=1e-12) -> float:
    mse = torch.mean((pred01 - gt01) ** 2).item()
    if mse < eps:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)

@torch.no_grad()
def validate_div2k_paired(model, val_lr_dir, val_hr_dir, scale=3, max_imgs=50, device="cuda"):
    import glob
    from PIL import Image
    import numpy as np

    lr_paths = sorted(glob.glob(os.path.join(val_lr_dir, f"*x{scale}.png")))
    if len(lr_paths) == 0:
        lr_paths = sorted(glob.glob(os.path.join(val_lr_dir, "*.png")))
    lr_paths = lr_paths[:max_imgs]

    psnrs = []
    model.eval()

    for lr_p in lr_paths:
        stem = os.path.splitext(os.path.basename(lr_p))[0].replace(f"x{scale}", "")
        hr_p = os.path.join(val_hr_dir, f"{stem}.png")
        if not os.path.exists(hr_p):
            continue

        lr = np.array(Image.open(lr_p).convert("RGB"), dtype=np.float32) / 255.0
        hr = np.array(Image.open(hr_p).convert("RGB"), dtype=np.float32) / 255.0

        lr_t = torch.from_numpy(lr).permute(2,0,1).unsqueeze(0).to(device)
        hr_t = torch.from_numpy(hr).permute(2,0,1).unsqueeze(0).to(device)

        sr = model(lr_t)
        H, W = hr_t.shape[-2:]
        sr = sr[..., :H, :W]
        sr = torch.clamp(sr, 0.0, 1.0)
        psnrs.append(psnr01(sr, hr_t))

    return float(sum(psnrs) / max(1, len(psnrs)))

def make_lr_schedule(opt, total_iters: int, milestones=(0.6, 0.9), gamma=0.5):
    ms = [int(total_iters * m) for m in milestones]
    def step(it):
        k = 0
        for m in ms:
            if it >= m:
                k += 1
        for g in opt.param_groups:
            g["lr"] = g["initial_lr"] * (gamma ** k)
    return step

def multiscale_loss(sr01, hr01, base="l1"):
    if base == "l2":
        l0 = F.mse_loss(sr01, hr01)
    elif base == "charb":
        l0 = charbonnier(sr01, hr01)
    else:
        l0 = F.l1_loss(sr01, hr01)

    # extra supervision ~x2 (downsample by 2/3)
    sr2 = F.interpolate(sr01, scale_factor=2/3, mode="bicubic", align_corners=False)
    hr2 = F.interpolate(hr01, scale_factor=2/3, mode="bicubic", align_corners=False)
    l2 = F.l1_loss(sr2, hr2)
    return l0 + 0.2 * l2

def infinite_loader(loader):
    for batch in cycle(loader):
        yield batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mambair_repo", type=str, required=True)
    ap.add_argument("--init_ckpt", type=str, required=True)

    ap.add_argument("--div2k_hr", type=str, required=True)
    ap.add_argument("--flickr2k_hr", type=str, required=True)
    ap.add_argument("--lsdir_hr", type=str, required=True)

    ap.add_argument("--div2k_val_lr", type=str, required=True)
    ap.add_argument("--div2k_val_hr", type=str, required=True)

    ap.add_argument("--out_dir", type=str, default="runs/teacher_mamba_x3")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--scale", type=int, default=3)

    ap.add_argument("--iters1", type=int, default=250000)
    ap.add_argument("--iters2", type=int, default=250000)
    ap.add_argument("--iters3", type=int, default=100000)

    ap.add_argument("--patch1", type=int, default=64)
    ap.add_argument("--patch2", type=int, default=128)
    ap.add_argument("--patch3", type=int, default=128)

    ap.add_argument("--batch1", type=int, default=8)
    ap.add_argument("--batch2", type=int, default=4)
    ap.add_argument("--batch3", type=int, default=4)

    ap.add_argument("--lr1", type=float, default=1e-4)
    ap.add_argument("--lr2", type=float, default=1e-5)
    ap.add_argument("--lr3", type=float, default=1e-5)

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--val_every", type=int, default=10000)
    ap.add_argument("--val_max", type=int, default=50)

    ap.add_argument("--w_div2k", type=float, default=0.34)
    ap.add_argument("--w_flickr2k", type=float, default=0.33)
    ap.add_argument("--w_lsdir", type=float, default=0.33)

    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    model = build_mambairv2_lightsr_x3(args.mambair_repo).to(device)

    ckpt = torch.load(args.init_ckpt, map_location="cpu")
    sd = _pick_state_dict(ckpt)
    model.load_state_dict(sd, strict=True)
    print("[init] loaded:", args.init_ckpt)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr1)
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    def run_stage(stage_name, iters, lr_patch, batch, base_lr, loss_base):
        for g in opt.param_groups:
            g["lr"] = base_lr
            g["initial_lr"] = base_lr

        ds = RandomHRMixOnTheFlyLR(
            [args.div2k_hr, args.flickr2k_hr, args.lsdir_hr],
            scale=args.scale,
            lr_patch=lr_patch,
            augment_on=True,
            num_samples=max(2000000, iters * batch),
            weights=[args.w_div2k, args.w_flickr2k, args.w_lsdir],
        )
        loader = DataLoader(ds, batch_size=batch, shuffle=True,
                            num_workers=args.num_workers, drop_last=True,
                            pin_memory=(device.type == "cuda"))
        it_loader = infinite_loader(loader)
        lr_step = make_lr_schedule(opt, total_iters=iters, milestones=(0.6, 0.9), gamma=0.5)

        best_psnr = -1e9
        best_path = os.path.join(args.out_dir, "teacher_best.pth")
        last_path = os.path.join(args.out_dir, "teacher_last.pth")

        pbar = tqdm(range(1, iters + 1), desc=f"[{stage_name}] patch={lr_patch} bs={batch} lr={base_lr}")
        model.train()

        for it in pbar:
            lr_step(it)
            lr01, hr01 = next(it_loader)
            lr01 = lr01.to(device)
            hr01 = hr01.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                sr01 = model(lr01)
                H, W = hr01.shape[-2], hr01.shape[-1]
                sr01 = sr01[..., :H, :W]
                sr01 = torch.clamp(sr01, 0.0, 1.0)
                loss = multiscale_loss(sr01, hr01, base=loss_base)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if it % 50 == 0:
                pbar.set_postfix(loss=float(loss.item()), lr=float(opt.param_groups[0]["lr"]))

            if (it % args.save_every == 0) or (it == iters):
                torch.save({"params": model.state_dict(), "stage": stage_name, "iter": it}, last_path)

            if (it % args.val_every == 0) or (it == iters):
                v = validate_div2k_paired(model, args.div2k_val_lr, args.div2k_val_hr,
                                          scale=args.scale, max_imgs=args.val_max, device=device)
                pbar.write(f"[val] {stage_name} iter={it} PSNR={v:.4f}")
                if v > best_psnr:
                    best_psnr = v
                    torch.save({"params": model.state_dict(), "stage": stage_name, "iter": it, "psnr": v}, best_path)
                    pbar.write(f"[best] saved teacher_best.pth PSNR={best_psnr:.4f}")

        pbar.close()

    run_stage("s1", args.iters1, args.patch1, args.batch1, args.lr1, loss_base="l1")
    run_stage("s2", args.iters2, args.patch2, args.batch2, args.lr2, loss_base="l2")
    run_stage("s3", args.iters3, args.patch3, args.batch3, args.lr3, loss_base="l1")

    print("[DONE] Teacher training finished.")
    print("Best ckpt:", os.path.join(args.out_dir, "teacher_best.pth"))

if __name__ == "__main__":
    main()