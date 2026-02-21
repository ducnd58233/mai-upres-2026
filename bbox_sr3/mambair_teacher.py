import sys
from typing import Dict, Any
import torch

def _pick_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for k in ["params", "state_dict", "model", "net", "params_ema"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise RuntimeError("Unknown checkpoint format")

def build_mambairv2_lightsr_x3(mambair_repo: str):
    sys.path.insert(0, mambair_repo)
    from basicsr.archs.mambairv2light_arch import MambaIRv2Light

    net = MambaIRv2Light(
        upscale=3,
        img_size=64,
        in_chans=3,
        img_range=1.0,
        embed_dim=48,
        d_state=8,
        depths=[5, 5, 5, 5],
        num_heads=[4, 4, 4, 4],
        window_size=16,
        inner_rank=32,
        num_tokens=64,
        convffn_kernel_size=5,
        mlp_ratio=1.0,
        upsampler="pixelshuffledirect",
        resi_connection="1conv",
    )
    return net

@torch.no_grad()
def load_teacher(mambair_repo: str, ckpt_path: str, device="cuda"):
    net = build_mambairv2_lightsr_x3(mambair_repo)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _pick_state_dict(ckpt)
    net.load_state_dict(sd, strict=True)
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    return net