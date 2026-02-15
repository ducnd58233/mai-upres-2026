from typing import Literal, Tuple

import torch
import torch.nn as nn


# -------------------------
# Helpers
# -------------------------
def conv3x3(in_ch: int, out_ch: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=bias)


def conv1x1(in_ch: int, out_ch: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=bias)


def pad_1x1_to_3x3_center(w_1x1: torch.Tensor) -> torch.Tensor:
    w_3x3 = w_1x1.new_zeros((w_1x1.size(0), w_1x1.size(1), 3, 3))
    w_3x3[:, :, 1, 1] = w_1x1[:, :, 0, 0]
    return w_3x3


@torch.no_grad()
def fuse_batchnorm_into_conv(
    W: torch.Tensor,  # (O,I,KH,KW)
    b: torch.Tensor,  # (O,)
    bn: nn.BatchNorm2d,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    inv_std = torch.rsqrt(var + eps)
    scale = gamma * inv_std

    W_fused = W * scale.view(-1, 1, 1, 1)
    b_fused = (b - mean) * scale + beta
    return W_fused, b_fused


# -------------------------
# Activations
# -------------------------
class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def make_activation(mode: Literal["none", "relu"]) -> nn.Module:
    if mode == "none":
        return Identity()
    if mode == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unknown activation mode: {mode}")


# -------------------------
# Tail clamp modes (paper-friendly)
# -------------------------
class Min255(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, max=255.0)


class MinClip(nn.Module):
    """min(ReLU(x),255) - SCSRN trick to reduce latency vs standalone ReLU+clip"""

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        return torch.clamp(x, max=255.0)


class Clamp0_255(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, 255.0)


def make_output_clamp(
    mode: Literal["min255", "minclip", "clamp_0_255", "none"]
) -> nn.Module:
    if mode == "min255":
        return Min255()
    if mode == "minclip":
        return MinClip()
    if mode == "clamp_0_255":
        return Clamp0_255()
    if mode == "none":
        return Identity()
    raise ValueError(f"Unknown clamp mode: {mode}")


# -------------------------
# RepConv (train: 1x1->3x3->1x1 + skip 1x1; deploy: single 3x3)
# -------------------------
class RepConv(nn.Module):
    def __init__(
        self,
        channels: int,
        deploy: bool = False,
        rep_use_bn: bool = False,
        act_mode: Literal["none", "relu"] = "none",
    ):
        super().__init__()
        self.c = channels
        self.deploy = deploy
        self.act = make_activation(act_mode)

        if deploy:
            self.reparam = conv3x3(channels, channels, bias=True)
            self.bn = Identity()
        else:
            self.conv1_a = conv1x1(channels, channels, bias=False)
            self.conv3 = conv3x3(channels, channels, bias=True)
            self.conv1_b = conv1x1(channels, channels, bias=True)
            self.conv1_s = conv1x1(channels, channels, bias=True)
            self.bn = nn.BatchNorm2d(channels) if rep_use_bn else Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.act(self.reparam(x))

        y_main = self.conv1_b(self.conv3(self.conv1_a(x)))
        y_skip = self.conv1_s(x)
        y = y_main + y_skip
        y = self.bn(y)
        y = self.act(y)
        return y

    @torch.no_grad()
    def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        W1 = self.conv1_a.weight.squeeze(-1).squeeze(-1)
        W3 = self.conv3.weight
        b3 = self.conv3.bias
        W2 = self.conv1_b.weight.squeeze(-1).squeeze(-1)
        b2 = self.conv1_b.bias

        tmp = torch.einsum("mnxy,ni->mixy", W3, W1)
        W_main = torch.einsum("om,mixy->oixy", W2, tmp)
        b_main = b2 + torch.matmul(W2, b3)

        W_skip = pad_1x1_to_3x3_center(self.conv1_s.weight)
        b_skip = self.conv1_s.bias

        W_eq = W_main + W_skip
        b_eq = b_main + b_skip

        if isinstance(self.bn, nn.BatchNorm2d):
            W_eq, b_eq = fuse_batchnorm_into_conv(W_eq, b_eq, self.bn)

        return W_eq, b_eq

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.deploy:
            return

        W, b = self.get_equivalent_kernel_bias()
        self.reparam = conv3x3(self.c, self.c, bias=True).to(
            device=W.device, dtype=W.dtype
        )
        self.reparam.weight.copy_(W)
        self.reparam.bias.copy_(b.to(W.dtype))

        del self.conv1_a, self.conv3, self.conv1_b, self.conv1_s
        self.bn = Identity()
        self.deploy = True


# -------------------------
# AntSR + SCSRN-style options
# -------------------------
ConcatHTR = Literal["3x3_3x3", "1x1_3x3", "1x1_1x1"]
SkipMode = Literal["add", "add1x1", "concat_lr", "concat_raw"]


def _make_trans_layers(
    in_ch: int, mid_ch: int, mode: ConcatHTR
) -> Tuple[nn.Module, nn.Module]:
    if mode == "3x3_3x3":
        return conv3x3(in_ch, mid_ch, True), conv3x3(mid_ch, mid_ch, True)
    if mode == "1x1_3x3":
        return conv1x1(in_ch, mid_ch, True), conv3x3(mid_ch, mid_ch, True)
    if mode == "1x1_1x1":
        return conv1x1(in_ch, mid_ch, True), conv1x1(mid_ch, mid_ch, True)
    raise ValueError(f"Unknown concat_htr: {mode}")


class AntSR(nn.Module):
    """
    AntSR x3 super-resolution:
      LR -> Conv3(3->C) -> RepConv xN -> (optional) global skip -> Conv3(C->27)
      -> out_clamp -> PixelShuffle(3)
    SCSRN-style skip_mode: concat_raw, concat_lr, add1x1, add.
    """

    def __init__(
        self,
        scale: int = 3,
        channels: int = 32,
        n_rep: int = 4,
        deploy: bool = False,
        rep_use_bn: bool = False,
        rep_act_mode: Literal["none", "relu"] = "none",
        out_clamp_mode: Literal["min255", "minclip", "clamp_0_255", "none"] = "minclip",
        skip_mode: SkipMode = "add",
        concat_htr: ConcatHTR = "3x3_3x3",
        use_global_add: bool = True,
    ):
        super().__init__()
        assert scale == 3, "This implementation targets x3 SR"
        self.scale = scale
        self.skip_mode = skip_mode
        self.concat_htr = concat_htr
        self.use_global_add = use_global_add

        self.conv_in = conv3x3(3, channels, bias=True)

        self.rep = nn.ModuleList(
            [
                RepConv(
                    channels,
                    deploy=deploy,
                    rep_use_bn=rep_use_bn,
                    act_mode=rep_act_mode,
                )
                for _ in range(n_rep)
            ]
        )

        if skip_mode in ("add1x1", "concat_lr"):
            self.lr_proj = conv1x1(3, channels, bias=True)
        else:
            self.lr_proj = Identity()

        if skip_mode == "concat_raw":
            self.htr1, self.htr2 = _make_trans_layers(
                channels + 3, channels, concat_htr
            )
        elif skip_mode == "concat_lr":
            self.htr1, self.htr2 = _make_trans_layers(
                channels * 2, channels, concat_htr
            )
        else:
            self.htr1 = Identity()
            self.htr2 = Identity()

        self.conv_out = conv3x3(channels, 3 * (scale * scale), bias=True)
        self.out_clamp = make_output_clamp(out_clamp_mode)
        self.ps = nn.PixelShuffle(scale)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        feat = self.conv_in(lr)
        x = feat
        for blk in self.rep:
            x = blk(x)

        if self.use_global_add:
            x = x + feat

        if self.skip_mode == "add1x1":
            x = x + self.lr_proj(lr)
        elif self.skip_mode == "concat_lr":
            lr_f = self.lr_proj(lr)
            x = torch.cat([x, lr_f], dim=1)
            x = self.htr2(self.htr1(x))
        elif self.skip_mode == "concat_raw":
            x = torch.cat([x, lr], dim=1)
            x = self.htr2(self.htr1(x))

        x = self.conv_out(x)
        x = self.out_clamp(x)
        return self.ps(x)

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        self.eval()
        for blk in self.rep:
            blk.switch_to_deploy()
