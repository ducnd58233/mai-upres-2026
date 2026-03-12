import os
import json
import torch
from torch.utils.data import DataLoader

from model_antsr import AntSR
from data_div2k_pairs import DIV2KPairX3, resolve_div2k_paths
from train_antsr_fp32 import load_ckpt_weights, validate_metrics_all

run_dir = "/home/namnguyen/projects/MobileAI/bbox_sr3/student/runs_mobileone_32_10_cleanfix_plus/balanced"
best_ckpt = os.path.join(run_dir, "ckpt_best_s2_fp32.pt")
run_cfg = os.path.join(run_dir, "run_config.json")
export_path = os.path.join(run_dir, "ckpt_best_s2_deploy.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(run_cfg, "r", encoding="utf-8") as f:
    rc = json.load(f)

args = rc["args"]
cfg = rc["extra"]["model_cfg"]

train_hr, train_lr, valid_hr, valid_lr = resolve_div2k_paths(args["data_root"], scale=3)
val_ds = DIV2KPairX3(valid_hr, valid_lr, train=False, lr_patch=128, augment=False, repeat=1)
val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=args.get("workers", 2),
    pin_memory=(device.type == "cuda"),
)

model = AntSR(deploy=False, **cfg).to(device)
load_ckpt_weights(model, best_ckpt, device, prefer_ema=True)
model.set_out_clamp_mode(args["out_clamp_export"])
model.eval()

m_float = validate_metrics_all(
    model,
    val_loader,
    device,
    scale=3,
    max_images=args.get("val_max", 0),
    shaves=(0, 3),
    report_ssim=False,
)
print("FLOAT pre-deploy:", m_float["psnr_sr_rgb_sh0"])

model.switch_to_deploy()
model.eval()

m_deploy = validate_metrics_all(
    model,
    val_loader,
    device,
    scale=3,
    max_images=args.get("val_max", 0),
    shaves=(0, 3),
    report_ssim=False,
)
print("DEPLOY no-recalib:", m_deploy["psnr_sr_rgb_sh0"])

export_cfg = dict(cfg)
export_cfg["out_clamp_mode"] = args["out_clamp_export"]

torch.save(
    {
        "model": model.state_dict(),
        "cfg": export_cfg,
    },
    export_path,
)
print("Saved:", export_path)