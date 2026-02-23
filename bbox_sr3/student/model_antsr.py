# # import torch
# # import torch.nn as nn
# # from typing import Tuple, Literal


# # # -------------------------
# # # Helpers
# # # -------------------------
# # def conv3x3(in_ch: int, out_ch: int, bias: bool = True) -> nn.Conv2d:
# #     return nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=bias)

# # def conv1x1(in_ch: int, out_ch: int, bias: bool = True) -> nn.Conv2d:
# #     return nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=bias)

# # def pad_1x1_to_3x3_center(w_1x1: torch.Tensor) -> torch.Tensor:
# #     w_3x3 = w_1x1.new_zeros((w_1x1.size(0), w_1x1.size(1), 3, 3))
# #     w_3x3[:, :, 1, 1] = w_1x1[:, :, 0, 0]
# #     return w_3x3

# # @torch.no_grad()
# # def fuse_batchnorm_into_conv(
# #     W: torch.Tensor,  # (O,I,KH,KW)
# #     b: torch.Tensor,  # (O,)
# #     bn: nn.BatchNorm2d,
# # ) -> Tuple[torch.Tensor, torch.Tensor]:
# #     gamma = bn.weight
# #     beta = bn.bias
# #     mean = bn.running_mean
# #     var = bn.running_var
# #     eps = bn.eps

# #     inv_std = torch.rsqrt(var + eps)
# #     scale = gamma * inv_std

# #     W_fused = W * scale.view(-1, 1, 1, 1)
# #     b_fused = (b - mean) * scale + beta
# #     return W_fused, b_fused


# # # -------------------------
# # # Activations
# # # -------------------------
# # class Identity(nn.Module):
# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         return x

# # def make_activation(mode: Literal["none", "relu"]) -> nn.Module:
# #     if mode == "none":
# #         return Identity()
# #     if mode == "relu":
# #         return nn.ReLU(inplace=True)
# #     raise ValueError(f"Unknown activation mode: {mode}")


# # # -------------------------
# # # Tail clamp modes (paper-friendly)
# # # -------------------------
# # class Min255(nn.Module):
# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         return torch.clamp(x, max=255.0)

# # class MinClip(nn.Module):
# #     """min(ReLU(x),255) - SCSRN trick to reduce latency vs standalone ReLU+clip"""
# #     def __init__(self):
# #         super().__init__()
# #         self.relu = nn.ReLU(inplace=True)

# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         x = self.relu(x)
# #         return torch.clamp(x, max=255.0)

# # class Clamp0_255(nn.Module):
# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         return torch.clamp(x, 0.0, 255.0)

# # def make_output_clamp(mode: Literal["min255", "minclip", "clamp_0_255", "none"]) -> nn.Module:
# #     if mode == "min255":
# #         return Min255()
# #     if mode == "minclip":
# #         return MinClip()
# #     if mode == "clamp_0_255":
# #         return Clamp0_255()
# #     if mode == "none":
# #         return Identity()
# #     raise ValueError(f"Unknown clamp mode: {mode}")


# # # -------------------------
# # # RepConv (train: 1x1->3x3->1x1 + skip 1x1; deploy: single 3x3)
# # # -------------------------
# # class RepConv(nn.Module):
# #     def __init__(
# #         self,
# #         channels: int,
# #         deploy: bool = False,
# #         rep_use_bn: bool = False,
# #         act_mode: Literal["none", "relu"] = "none",
# #     ):
# #         super().__init__()
# #         self.c = channels
# #         self.deploy = deploy
# #         self.act = make_activation(act_mode)

# #         if deploy:
# #             self.reparam = conv3x3(channels, channels, bias=True)
# #             self.bn = Identity()
# #         else:
# #             # conv1_a bias=False helps exact math for reparam
# #             self.conv1_a = conv1x1(channels, channels, bias=False)
# #             self.conv3   = conv3x3(channels, channels, bias=True)
# #             self.conv1_b = conv1x1(channels, channels, bias=True)
# #             self.conv1_s = conv1x1(channels, channels, bias=True)
# #             self.bn = nn.BatchNorm2d(channels) if rep_use_bn else Identity()

# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         if self.deploy:
# #             return self.act(self.reparam(x))

# #         y_main = self.conv1_b(self.conv3(self.conv1_a(x)))
# #         y_skip = self.conv1_s(x)
# #         y = y_main + y_skip
# #         y = self.bn(y)
# #         y = self.act(y)
# #         return y

# #     @torch.no_grad()
# #     def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
# #         # main: conv1_a -> conv3 -> conv1_b
# #         W1 = self.conv1_a.weight.squeeze(-1).squeeze(-1)   # (C,C)
# #         W3 = self.conv3.weight                              # (C,C,3,3)
# #         b3 = self.conv3.bias                                # (C,)
# #         W2 = self.conv1_b.weight.squeeze(-1).squeeze(-1)   # (C,C)
# #         b2 = self.conv1_b.bias                              # (C,)

# #         tmp = torch.einsum("mnxy,ni->mixy", W3, W1)         # (C,C,3,3)
# #         W_main = torch.einsum("om,mixy->oixy", W2, tmp)     # (C,C,3,3)
# #         b_main = b2 + torch.matmul(W2, b3)                  # (C,)

# #         # skip: conv1_s -> pad to center of 3x3
# #         W_skip = pad_1x1_to_3x3_center(self.conv1_s.weight) # (C,C,3,3)
# #         b_skip = self.conv1_s.bias                          # (C,)

# #         W_eq = W_main + W_skip
# #         b_eq = b_main + b_skip

# #         if isinstance(self.bn, nn.BatchNorm2d):
# #             W_eq, b_eq = fuse_batchnorm_into_conv(W_eq, b_eq, self.bn)

# #         return W_eq, b_eq

# #     @torch.no_grad()
# #     def switch_to_deploy(self) -> None:
# #         if self.deploy:
# #             return

# #         W, b = self.get_equivalent_kernel_bias()
# #         self.reparam = conv3x3(self.c, self.c, bias=True).to(device=W.device, dtype=W.dtype)
# #         self.reparam.weight.copy_(W)
# #         self.reparam.bias.copy_(b.to(W.dtype))

# #         del self.conv1_a, self.conv3, self.conv1_b, self.conv1_s
# #         self.bn = Identity()
# #         self.deploy = True


# # # -------------------------
# # # AntSR + SCSRN-style options (best-score friendly)
# # # -------------------------
# # ConcatHTR = Literal["3x3_3x3", "1x1_3x3", "1x1_1x1"]
# # SkipMode  = Literal["add", "add1x1", "concat_lr", "concat_raw"]

# # def _make_trans_layers(in_ch: int, mid_ch: int, mode: ConcatHTR) -> Tuple[nn.Module, nn.Module]:
# #     """
# #     Two transition layers after concat (HTR, HTR) like SCSRN:
# #       3x3_3x3 : best PSNR, slower
# #       1x1_3x3 : balanced
# #       1x1_1x1 : fastest, may reduce PSNR
# #     """
# #     if mode == "3x3_3x3":
# #         return conv3x3(in_ch, mid_ch, True), conv3x3(mid_ch, mid_ch, True)
# #     if mode == "1x1_3x3":
# #         return conv1x1(in_ch, mid_ch, True), conv3x3(mid_ch, mid_ch, True)
# #     if mode == "1x1_1x1":
# #         return conv1x1(in_ch, mid_ch, True), conv1x1(mid_ch, mid_ch, True)
# #     raise ValueError(f"Unknown concat_htr: {mode}")

# # class AntSR(nn.Module):
# #     """
# #     Base (AntSR):
# #       LR -> Conv3(3->C)
# #          -> RepConv xN
# #          -> (optional) global skip add with feat
# #          -> Conv3(C->27)
# #          -> out_clamp (minclip/min255)
# #          -> PixelShuffle(3)

# #     SCSRN-style add-ons:
# #       skip_mode="concat_raw": concat([x, lr]) -> HTR1 -> HTR2  (recommended for INT8 PSNR)
# #       skip_mode="concat_lr" : project lr -> C then concat([x, lr_proj]) -> HTR1 -> HTR2
# #       skip_mode="add1x1"    : x = x + lr_proj(lr)   (cheapest)
# #     """

# #     def __init__(
# #         self,
# #         scale: int = 3,
# #         channels: int = 32,
# #         n_rep: int = 4,
# #         deploy: bool = False,
# #         rep_use_bn: bool = False,
# #         rep_act_mode: Literal["none", "relu"] = "none",
# #         out_clamp_mode: Literal["min255", "minclip", "clamp_0_255", "none"] = "minclip",
# #         skip_mode: SkipMode = "add",
# #         concat_htr: ConcatHTR = "3x3_3x3",
# #         use_global_add: bool = True,
# #     ):
# #         super().__init__()
# #         assert scale == 3, "This implementation targets x3 SR"
# #         self.scale = scale
# #         self.skip_mode = skip_mode
# #         self.concat_htr = concat_htr
# #         self.use_global_add = use_global_add

# #         self.conv_in = conv3x3(3, channels, bias=True)

# #         self.rep = nn.ModuleList([
# #             RepConv(channels, deploy=deploy, rep_use_bn=rep_use_bn, act_mode=rep_act_mode)
# #             for _ in range(n_rep)
# #         ])

# #         # LR projection only used by add1x1 / concat_lr
# #         if skip_mode in ("add1x1", "concat_lr"):
# #             self.lr_proj = conv1x1(3, channels, bias=True)
# #         else:
# #             self.lr_proj = Identity()

# #         # Transition layers (HTR1, HTR2) for concat modes
# #         if skip_mode == "concat_raw":
# #             self.htr1, self.htr2 = _make_trans_layers(channels + 3, channels, concat_htr)
# #         elif skip_mode == "concat_lr":
# #             self.htr1, self.htr2 = _make_trans_layers(channels * 2, channels, concat_htr)
# #         else:
# #             self.htr1 = Identity()
# #             self.htr2 = Identity()

# #         self.conv_out = conv3x3(channels, 3 * (scale * scale), bias=True)  # 27
# #         self.out_clamp = make_output_clamp(out_clamp_mode)
# #         self.ps = nn.PixelShuffle(scale)

# #     def forward(self, lr: torch.Tensor) -> torch.Tensor:
# #         feat = self.conv_in(lr)
# #         x = feat
# #         for blk in self.rep:
# #             x = blk(x)

# #         if self.use_global_add:
# #             x = x + feat

# #         if self.skip_mode == "add1x1":
# #             x = x + self.lr_proj(lr)

# #         elif self.skip_mode == "concat_lr":
# #             lr_f = self.lr_proj(lr)
# #             x = torch.cat([x, lr_f], dim=1)
# #             x = self.htr2(self.htr1(x))

# #         elif self.skip_mode == "concat_raw":
# #             # IMPORTANT: concat RAW LR (3ch) = best INT8 PSNR per SCSRN spirit
# #             x = torch.cat([x, lr], dim=1)
# #             x = self.htr2(self.htr1(x))

# #         x = self.conv_out(x)
# #         x = self.out_clamp(x)
# #         return self.ps(x)

# #     @torch.no_grad()
# #     def switch_to_deploy(self) -> None:
# #         self.eval()
# #         for blk in self.rep:
# #             blk.switch_to_deploy()





# import torch
# import torch.nn as nn
# from typing import Tuple, Literal, Optional


# # -------------------------
# # Helpers
# # -------------------------
# def conv3x3(in_ch: int, out_ch: int, bias: bool = True) -> nn.Conv2d:
#     return nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=bias)

# def conv1x1(in_ch: int, out_ch: int, bias: bool = True) -> nn.Conv2d:
#     return nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=bias)

# def pad_1x1_to_3x3_center(w_1x1: torch.Tensor) -> torch.Tensor:
#     w_3x3 = w_1x1.new_zeros((w_1x1.size(0), w_1x1.size(1), 3, 3))
#     w_3x3[:, :, 1, 1] = w_1x1[:, :, 0, 0]
#     return w_3x3

# @torch.no_grad()
# def fuse_batchnorm_into_conv(
#     W: torch.Tensor,  # (O,I,KH,KW)
#     b: torch.Tensor,  # (O,)
#     bn: nn.BatchNorm2d,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     gamma = bn.weight
#     beta = bn.bias
#     mean = bn.running_mean
#     var = bn.running_var
#     eps = bn.eps

#     inv_std = torch.rsqrt(var + eps)
#     scale = gamma * inv_std

#     W_fused = W * scale.view(-1, 1, 1, 1)
#     b_fused = (b - mean) * scale + beta
#     return W_fused, b_fused


# # -------------------------
# # Activations
# # -------------------------
# class Identity(nn.Module):
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x

# def make_activation(mode: Literal["none", "relu"]) -> nn.Module:
#     if mode == "none":
#         return Identity()
#     if mode == "relu":
#         return nn.ReLU(inplace=True)
#     raise ValueError(f"Unknown activation mode: {mode}")


# # -------------------------
# # Tail clamp modes (export-friendly)
# # -------------------------
# class Min255(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.register_buffer("_mx", torch.tensor(255.0))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return torch.minimum(x, self._mx)

# class MinClip(nn.Module):
#     """min(ReLU(x),255) - SCSRN trick (often faster than standalone ReLU+clip)"""
#     def __init__(self):
#         super().__init__()
#         self.relu = nn.ReLU(inplace=True)
#         self.register_buffer("_mx", torch.tensor(255.0))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.relu(x)
#         return torch.minimum(x, self._mx)

# class Clamp0_255(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.register_buffer("_mn", torch.tensor(0.0))
#         self.register_buffer("_mx", torch.tensor(255.0))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = torch.maximum(x, self._mn)
#         return torch.minimum(x, self._mx)

# def make_output_clamp(mode: Literal["min255", "minclip", "clamp_0_255", "none"]) -> nn.Module:
#     if mode == "min255":
#         return Min255()
#     if mode == "minclip":
#         return MinClip()
#     if mode == "clamp_0_255":
#         return Clamp0_255()
#     if mode == "none":
#         return Identity()
#     raise ValueError(f"Unknown clamp mode: {mode}")


# # -------------------------
# # RepConv (train: 1x1->3x3->1x1 + skip 1x1; deploy: single 3x3)
# # -------------------------
# class RepConv(nn.Module):
#     def __init__(
#         self,
#         channels: int,
#         deploy: bool = False,
#         rep_use_bn: bool = False,
#         act_mode: Literal["none", "relu"] = "none",
#     ):
#         super().__init__()
#         self.c = channels
#         self.deploy = deploy
#         self.act = make_activation(act_mode)

#         if deploy:
#             self.reparam = conv3x3(channels, channels, bias=True)
#             self.bn = Identity()
#         else:
#             # conv1_a bias=False helps exact math for reparam
#             self.conv1_a = conv1x1(channels, channels, bias=False)
#             self.conv3   = conv3x3(channels, channels, bias=True)
#             self.conv1_b = conv1x1(channels, channels, bias=True)
#             self.conv1_s = conv1x1(channels, channels, bias=True)
#             self.bn = nn.BatchNorm2d(channels) if rep_use_bn else Identity()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.deploy:
#             return self.act(self.reparam(x))

#         y_main = self.conv1_b(self.conv3(self.conv1_a(x)))
#         y_skip = self.conv1_s(x)
#         y = y_main + y_skip
#         y = self.bn(y)
#         y = self.act(y)
#         return y

#     @torch.no_grad()
#     def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
#         # main: conv1_a -> conv3 -> conv1_b
#         W1 = self.conv1_a.weight.squeeze(-1).squeeze(-1)   # (C,C)
#         W3 = self.conv3.weight                              # (C,C,3,3)
#         b3 = self.conv3.bias                                # (C,)
#         W2 = self.conv1_b.weight.squeeze(-1).squeeze(-1)   # (C,C)
#         b2 = self.conv1_b.bias                              # (C,)

#         tmp = torch.einsum("mnxy,ni->mixy", W3, W1)         # (C,C,3,3)
#         W_main = torch.einsum("om,mixy->oixy", W2, tmp)     # (C,C,3,3)
#         b_main = b2 + torch.matmul(W2, b3)                  # (C,)

#         # skip: conv1_s -> pad to center of 3x3
#         W_skip = pad_1x1_to_3x3_center(self.conv1_s.weight) # (C,C,3,3)
#         b_skip = self.conv1_s.bias                          # (C,)

#         W_eq = W_main + W_skip
#         b_eq = b_main + b_skip

#         if isinstance(self.bn, nn.BatchNorm2d):
#             W_eq, b_eq = fuse_batchnorm_into_conv(W_eq, b_eq, self.bn)

#         return W_eq, b_eq

#     @torch.no_grad()
#     def switch_to_deploy(self) -> None:
#         if self.deploy:
#             return

#         W, b = self.get_equivalent_kernel_bias()
#         self.reparam = conv3x3(self.c, self.c, bias=True).to(device=W.device, dtype=W.dtype)
#         self.reparam.weight.copy_(W)
#         self.reparam.bias.copy_(b.to(W.dtype))

#         del self.conv1_a, self.conv3, self.conv1_b, self.conv1_s
#         self.bn = Identity()
#         self.deploy = True


# # -------------------------
# # AntSR + SCSRN-style options
# # -------------------------
# ConcatHTR = Literal["3x3_3x3", "1x1_3x3", "1x1_1x1"]
# SkipMode  = Literal["add", "add1x1", "concat_lr", "concat_raw"]

# def _make_trans_layers(in_ch: int, mid_ch: int, mode: ConcatHTR) -> Tuple[nn.Module, nn.Module]:
#     """
#     Two transition layers after concat (HTR, HTR) like SCSRN:
#       3x3_3x3 : best PSNR, slower
#       1x1_3x3 : balanced
#       1x1_1x1 : fastest, may reduce PSNR
#     """
#     if mode == "3x3_3x3":
#         return conv3x3(in_ch, mid_ch, True), conv3x3(mid_ch, mid_ch, True)
#     if mode == "1x1_3x3":
#         return conv1x1(in_ch, mid_ch, True), conv3x3(mid_ch, mid_ch, True)
#     if mode == "1x1_1x1":
#         return conv1x1(in_ch, mid_ch, True), conv1x1(mid_ch, mid_ch, True)
#     raise ValueError(f"Unknown concat_htr: {mode}")

# class AntSR(nn.Module):
#     """
#     Base (AntSR):
#       LR -> Conv3(3->C)
#          -> RepConv xN
#          -> (optional) global skip add with feat
#          -> optional skip_mode (add/concat)
#          -> Conv3(C->27)
#          -> out_clamp (minclip/min255)
#          -> PixelShuffle(3)

#     SCSRN-style add-ons:
#       skip_mode="concat_raw": concat([x, lr]) -> HTR1 -> HTR2
#       skip_mode="concat_lr" : project lr -> C then concat([x, lr_proj]) -> HTR1 -> HTR2
#       skip_mode="add1x1"    : x = x + lr_proj(lr)
#     """

#     def __init__(
#         self,
#         scale: int = 3,
#         channels: int = 32,
#         n_rep: int = 4,
#         deploy: bool = False,
#         rep_use_bn: bool = False,
#         rep_act_mode: Literal["none", "relu"] = "none",
#         out_clamp_mode: Literal["min255", "minclip", "clamp_0_255", "none"] = "minclip",
#         skip_mode: SkipMode = "add",
#         concat_htr: ConcatHTR = "3x3_3x3",
#         use_global_add: bool = True,
#         raw_affine: bool = False,  # << NEW: tiny affine on raw LR before concat_raw
#     ):
#         super().__init__()
#         assert scale == 3, "This implementation targets x3 SR"
#         self.scale = scale
#         self.skip_mode = skip_mode
#         self.concat_htr = concat_htr
#         self.use_global_add = use_global_add
#         self.raw_affine = raw_affine

#         self.conv_in = conv3x3(3, channels, bias=True)

#         self.rep = nn.ModuleList([
#             RepConv(channels, deploy=deploy, rep_use_bn=rep_use_bn, act_mode=rep_act_mode)
#             for _ in range(n_rep)
#         ])

#         # LR projection only used by add1x1 / concat_lr
#         if skip_mode in ("add1x1", "concat_lr"):
#             self.lr_proj = conv1x1(3, channels, bias=True)
#         else:
#             self.lr_proj = Identity()

#         # raw affine only for concat_raw (optional)
#         if skip_mode == "concat_raw" and raw_affine:
#             self.raw_scale = nn.Parameter(torch.ones(1, 3, 1, 1))
#             self.raw_bias  = nn.Parameter(torch.zeros(1, 3, 1, 1))
#         else:
#             self.raw_scale = None
#             self.raw_bias  = None

#         # Transition layers (HTR1, HTR2) for concat modes
#         if skip_mode == "concat_raw":
#             self.htr1, self.htr2 = _make_trans_layers(channels + 3, channels, concat_htr)
#         elif skip_mode == "concat_lr":
#             self.htr1, self.htr2 = _make_trans_layers(channels * 2, channels, concat_htr)
#         else:
#             self.htr1 = Identity()
#             self.htr2 = Identity()

#         self.conv_out = conv3x3(channels, 3 * (scale * scale), bias=True)  # 27
#         self.out_clamp = make_output_clamp(out_clamp_mode)
#         self.ps = nn.PixelShuffle(scale)

#     def forward(self, lr: torch.Tensor) -> torch.Tensor:
#         feat = self.conv_in(lr)
#         x = feat
#         for blk in self.rep:
#             x = blk(x)

#         if self.use_global_add:
#             x = x + feat

#         if self.skip_mode == "add1x1":
#             x = x + self.lr_proj(lr)

#         elif self.skip_mode == "concat_lr":
#             lr_f = self.lr_proj(lr)
#             x = torch.cat([x, lr_f], dim=1)
#             x = self.htr2(self.htr1(x))

#         elif self.skip_mode == "concat_raw":
#             lr_raw = lr
#             if self.raw_scale is not None:
#                 lr_raw = lr_raw * self.raw_scale + self.raw_bias
#             x = torch.cat([x, lr_raw], dim=1)
#             x = self.htr2(self.htr1(x))

#         x = self.conv_out(x)
#         x = self.out_clamp(x)
#         return self.ps(x)

#     @torch.no_grad()
#     def switch_to_deploy(self) -> None:
#         self.eval()
#         for blk in self.rep:
#             blk.switch_to_deploy()












import torch
import torch.nn as nn
from typing import Tuple, Literal


# -------------------------
# Helpers
# -------------------------
def conv3x3(in_ch: int, out_ch: int, bias: bool = True, groups: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=bias, groups=groups)

def conv1x1(in_ch: int, out_ch: int, bias: bool = True, groups: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=bias, groups=groups)

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
    beta  = bn.bias
    mean  = bn.running_mean
    var   = bn.running_var
    eps   = bn.eps

    inv_std = torch.rsqrt(var + eps)
    scale = gamma * inv_std

    W_fused = W * scale.view(-1, 1, 1, 1)
    b_fused = (b - mean) * scale + beta
    return W_fused, b_fused

def identity_3x3_kernel(C: int, device, dtype) -> torch.Tensor:
    W = torch.zeros((C, C, 3, 3), device=device, dtype=dtype)
    idx = torch.arange(C, device=device)
    W[idx, idx, 1, 1] = 1.0
    return W


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
# Tail clamp modes
# -------------------------
class Min255(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, max=255.0)

class MinClip(nn.Module):
    """min(ReLU(x),255)"""
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        return torch.clamp(x, max=255.0)

class Clamp0_255(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, 255.0)

def make_output_clamp(mode: Literal["min255", "minclip", "clamp_0_255", "none"]) -> nn.Module:
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
# RepConv (bottleneck style)
# -------------------------
class RepConv(nn.Module):
    """
    train: 1x1->3x3->1x1 + skip 1x1 (+ optional BN) -> act
    deploy: single 3x3 (+ act)
    """
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
            self.conv3   = conv3x3(channels, channels, bias=True)
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
        W1 = self.conv1_a.weight.squeeze(-1).squeeze(-1)   # (C,C)
        W3 = self.conv3.weight                              # (C,C,3,3)
        b3 = self.conv3.bias                                # (C,)
        W2 = self.conv1_b.weight.squeeze(-1).squeeze(-1)   # (C,C)
        b2 = self.conv1_b.bias                              # (C,)

        tmp = torch.einsum("mnxy,ni->mixy", W3, W1)         # (C,C,3,3)
        W_main = torch.einsum("om,mixy->oixy", W2, tmp)     # (C,C,3,3)
        b_main = b2 + torch.matmul(W2, b3)                  # (C,)

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
        self.reparam = conv3x3(self.c, self.c, bias=True).to(device=W.device, dtype=W.dtype)
        self.reparam.weight.copy_(W)
        self.reparam.bias.copy_(b.to(W.dtype))

        del self.conv1_a, self.conv3, self.conv1_b, self.conv1_s
        self.bn = Identity()
        self.deploy = True


# -------------------------
# MobileOne-style multi-branch rep block
# -------------------------
class _ConvBN(nn.Module):
    def __init__(self, conv: nn.Conv2d, use_bn: bool):
        super().__init__()
        self.conv = conv
        self.bn = nn.BatchNorm2d(conv.out_channels) if use_bn else Identity()

    def forward(self, x):
        return self.bn(self.conv(x))

    @torch.no_grad()
    def fuse(self) -> Tuple[torch.Tensor, torch.Tensor]:
        W = self.conv.weight
        if self.conv.bias is None:
            b = torch.zeros((W.size(0),), device=W.device, dtype=W.dtype)
        else:
            b = self.conv.bias
        if isinstance(self.bn, nn.BatchNorm2d):
            W, b = fuse_batchnorm_into_conv(W, b, self.bn)
        return W, b


class MobileOneRepBlock(nn.Module):
    """
    Train: sum of branches:
      - K x (3x3 conv + optional BN)
      - optional 1x1 conv branch (+ optional BN)
      - optional identity branch (+ optional BN)
    Deploy: fuse -> single 3x3 conv
    """
    def __init__(
        self,
        channels: int,
        deploy: bool = False,
        num_3x3_branches: int = 2,
        use_1x1: bool = True,
        use_identity: bool = True,
        rep_use_bn: bool = True,
        act_mode: Literal["none", "relu"] = "relu",
    ):
        super().__init__()
        self.c = channels
        self.deploy = deploy
        self.act = make_activation(act_mode)

        self.num_3x3 = int(num_3x3_branches)
        self.use_1x1 = bool(use_1x1)
        self.use_id  = bool(use_identity)
        self.use_bn  = bool(rep_use_bn)

        if deploy:
            self.reparam = conv3x3(channels, channels, bias=True)
        else:
            bias = not self.use_bn

            self.branches_3x3 = nn.ModuleList([
                _ConvBN(conv3x3(channels, channels, bias=bias), use_bn=self.use_bn)
                for _ in range(self.num_3x3)
            ])

            self.branch_1x1 = None
            if self.use_1x1:
                self.branch_1x1 = _ConvBN(conv1x1(channels, channels, bias=bias), use_bn=self.use_bn)

            self.id_bn = None
            if self.use_id:
                self.id_bn = nn.BatchNorm2d(channels) if self.use_bn else Identity()


    
    def forward(self, x):
        if self.deploy:
            return self.act(self.reparam(x))

        y = self.branches_3x3[0](x)
        for br in self.branches_3x3[1:]:
            y = y + br(x)
        if self.branch_1x1 is not None:
            y = y + self.branch_1x1(x)
        if self.id_bn is not None:
            y = y + self.id_bn(x)
        return self.act(y)


    @torch.no_grad()
    def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        W_sum = torch.zeros((self.c, self.c, 3, 3), device=device, dtype=dtype)
        b_sum = torch.zeros((self.c,), device=device, dtype=dtype)

        for br in self.branches_3x3:
            W, b = br.fuse()
            W_sum += W
            b_sum += b

        if self.branch_1x1 is not None:
            W1, b1 = self.branch_1x1.fuse()  # (C,C,1,1)
            W_sum += pad_1x1_to_3x3_center(W1)
            b_sum += b1

        if self.id_bn is not None:
            W_id = identity_3x3_kernel(self.c, device=device, dtype=dtype)
            b_id = torch.zeros((self.c,), device=device, dtype=dtype)
            if isinstance(self.id_bn, nn.BatchNorm2d):
                W_id, b_id = fuse_batchnorm_into_conv(W_id, b_id, self.id_bn)
            W_sum += W_id
            b_sum += b_id

        return W_sum, b_sum

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        W, b = self.get_equivalent_kernel_bias()

        self.reparam = conv3x3(self.c, self.c, bias=True).to(device=W.device, dtype=W.dtype)
        self.reparam.weight.copy_(W)
        self.reparam.bias.copy_(b.to(W.dtype))

        del self.branches_3x3
        if hasattr(self, "branch_1x1"):
            del self.branch_1x1
        if hasattr(self, "id_bn"):
            del self.id_bn

        self.deploy = True


# -------------------------
# RepDW block (speed candidate, no fusion)
# -------------------------
class RepDWBlock(nn.Module):
    def __init__(self, channels: int, act_mode: Literal["none", "relu"] = "relu"):
        super().__init__()
        self.dw = conv3x3(channels, channels, bias=True, groups=channels)
        self.pw = conv1x1(channels, channels, bias=True)
        self.act = make_activation(act_mode)

    def forward(self, x):
        y = self.pw(self.dw(x))
        y = y + x
        return self.act(y)

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        return


# -------------------------
# AntSR + options
# -------------------------
ConcatHTR = Literal["3x3_3x3", "1x1_3x3", "1x1_1x1"]
SkipMode  = Literal["add", "add1x1", "concat_lr", "concat_raw"]
RepType   = Literal["repconv", "mobileone", "repdw"]

def _make_trans_layers(in_ch: int, mid_ch: int, mode: ConcatHTR) -> Tuple[nn.Module, nn.Module]:
    if mode == "3x3_3x3":
        return conv3x3(in_ch, mid_ch, True), conv3x3(mid_ch, mid_ch, True)
    if mode == "1x1_3x3":
        return conv1x1(in_ch, mid_ch, True), conv3x3(mid_ch, mid_ch, True)
    if mode == "1x1_1x1":
        return conv1x1(in_ch, mid_ch, True), conv1x1(mid_ch, mid_ch, True)
    raise ValueError(f"Unknown concat_htr: {mode}")

class AntSR(nn.Module):
    def __init__(
        self,
        scale: int = 3,
        channels: int = 32,
        n_rep: int = 4,
        deploy: bool = False,

        rep_type: RepType = "repconv",

        rep_use_bn: bool = False,
        rep_act_mode: Literal["none", "relu"] = "none",

        mo_branches: int = 2,
        mo_use_1x1: bool = True,
        mo_use_identity: bool = True,

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
        self.rep_type = rep_type

        self.conv_in = conv3x3(3, channels, bias=True)

        if rep_type == "repconv":
            blocks = [
                RepConv(channels, deploy=deploy, rep_use_bn=rep_use_bn, act_mode=rep_act_mode)
                for _ in range(n_rep)
            ]
        elif rep_type == "mobileone":
            act = "relu" if rep_act_mode == "none" else rep_act_mode
            blocks = [
                MobileOneRepBlock(
                    channels,
                    deploy=deploy,
                    num_3x3_branches=mo_branches,
                    use_1x1=mo_use_1x1,
                    use_identity=mo_use_identity,
                    rep_use_bn=rep_use_bn,
                    act_mode=act,
                )
                for _ in range(n_rep)
            ]
        elif rep_type == "repdw":
            act = "relu" if rep_act_mode == "none" else rep_act_mode
            blocks = [RepDWBlock(channels, act_mode=act) for _ in range(n_rep)]
        else:
            raise ValueError(f"Unknown rep_type: {rep_type}")

        self.rep = nn.ModuleList(blocks)

        if skip_mode in ("add1x1", "concat_lr"):
            self.lr_proj = conv1x1(3, channels, bias=True)
        else:
            self.lr_proj = Identity()

        if skip_mode == "concat_raw":
            self.htr1, self.htr2 = _make_trans_layers(channels + 3, channels, concat_htr)
        elif skip_mode == "concat_lr":
            self.htr1, self.htr2 = _make_trans_layers(channels * 2, channels, concat_htr)
        else:
            self.htr1 = Identity()
            self.htr2 = Identity()

        self.conv_out = conv3x3(channels, 3 * (scale * scale), bias=True)  # 27
        self.out_clamp = make_output_clamp(out_clamp_mode)
        self.ps = nn.PixelShuffle(scale)

    def set_out_clamp_mode(self, mode: Literal["min255", "minclip", "clamp_0_255", "none"]):
        self.out_clamp = make_output_clamp(mode)

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
            if hasattr(blk, "switch_to_deploy"):
                blk.switch_to_deploy()

