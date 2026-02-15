import math
from typing import Any, Dict

import torch


def psnr_255(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-12) -> float:
    mse = torch.mean((pred - gt) ** 2).item()
    if mse < eps:
        return 99.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)


_DCT_CACHE: Dict[Any, Dict[str, torch.Tensor]] = {}


def _get_dct_cache(
    N: int, device: torch.device, dtype: torch.dtype
) -> Dict[str, torch.Tensor]:
    key = (N, device.type, device.index if device.type == "cuda" else -1, dtype)
    if key in _DCT_CACHE:
        return _DCT_CACHE[key]

    even_idx = torch.arange(0, N, 2, device=device)
    odd_idx = torch.arange(N - 1, -1, -2, device=device)
    cplx_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    k = torch.arange(N, device=device, dtype=dtype)
    W = torch.exp(-1j * math.pi * k / (2.0 * N)).to(cplx_dtype)

    _DCT_CACHE[key] = {"even": even_idx, "odd": odd_idx, "W": W}
    return _DCT_CACHE[key]


def _dct_1d(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    N = x.size(dim)
    cache = _get_dct_cache(N, x.device, x.dtype)
    even = x.index_select(dim, cache["even"])
    odd = x.index_select(dim, cache["odd"])
    v = torch.cat([even, odd], dim=dim)
    V = torch.fft.fft(v, dim=dim)
    return (V * cache["W"]).real * 2.0


def dct_2d(x: torch.Tensor) -> torch.Tensor:
    x = _dct_1d(x, dim=-1)
    x = _dct_1d(x, dim=-2)
    return x


def dct_l1_loss(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(dct_2d(sr) - dct_2d(hr)))
