from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AntSRTrainConfig:
    """AntSR 3-stage training configuration. Reused by training and CLI."""

    data_root: str
    out_dir: str
    seed: int = 1234
    batch: int = 16
    workers: int = 4
    val_max: int = 0
    val_every: int = 1
    # Model
    channels: int = 32
    n_rep: int = 4
    skip_mode: str = "concat_raw"
    concat_htr: str = "3x3_3x3"
    use_global_add: bool = True
    rep_use_bn: bool = False
    rep_act_mode: str = "none"
    out_clamp_mode: str = "minclip"
    # Stages
    epochs1: int = 800
    epochs2: int = 200
    epochs3: int = 300
    lr1: float = 1e-3
    lr2: float = 2e-5
    lr3: float = 1e-5
    patch1: int = 128
    patch2: int = 128
    patch3: int = 128
    s1_loss: str = "l1"
    s2_loss: str = "l2"
    s3_loss: str = "dct"
    scheduler1: str = "cos_warmup"
    scheduler2: str = "step_halve"
    scheduler3: str = "step_halve"
    channel_shuffle_s2s3: bool = True
    grad_clip: float = 0.0
    # QAT
    qat: bool = True
    qat_backend: str = "qnnpack"
    deploy_before_qat: bool = True
    weight_clipping: bool = True
    wc_other: float = 2.0
    wc_rep: float = 3.0
    init_ckpt: str | None = None
    preset: str = "balanced"
