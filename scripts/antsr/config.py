from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from data.constants import WEIGHT_CLIP_OTHER_DEFAULT, WEIGHT_CLIP_REP_DEFAULT
from models.antsr.model import ConcatHTR, SkipMode

OutClampMode = Literal["min255", "minclip", "clamp_0_255", "none"]
ActMode = Literal["none", "relu"]
LossMode = Literal["l1", "l2", "dct"]
SchedulerType = Literal["none", "cos_warmup", "step_halve"]


@dataclass(frozen=True)
class AntSRModelConfig:
    """Configuration for AntSR architecture."""

    scale: int = 3
    channels: int = 32
    n_rep: int = 4
    deploy: bool = False
    rep_use_bn: bool = False
    rep_act_mode: ActMode = "none"
    out_clamp_mode: OutClampMode = "minclip"
    skip_mode: SkipMode = "concat_raw"
    concat_htr: ConcatHTR = "3x3_3x3"
    use_global_add: bool = True

    def to_kwargs(self) -> dict:
        """For passing to AntSR(**kwargs)."""
        return {
            "scale": self.scale,
            "channels": self.channels,
            "n_rep": self.n_rep,
            "deploy": self.deploy,
            "rep_use_bn": self.rep_use_bn,
            "rep_act_mode": self.rep_act_mode,
            "out_clamp_mode": self.out_clamp_mode,
            "skip_mode": self.skip_mode,
            "concat_htr": self.concat_htr,
            "use_global_add": self.use_global_add,
        }


@dataclass
class TrainStageConfig:
    """Per-stage training config."""

    epochs: int
    base_lr: float
    lr_patch: int
    loss_mode: LossMode
    scheduler_type: SchedulerType
    weight_clip: bool = False
    wc_other: float = WEIGHT_CLIP_OTHER_DEFAULT
    wc_rep: float = WEIGHT_CLIP_REP_DEFAULT
    stage_tag: str = "stage"


@dataclass
class Div2kPaths:
    """Resolved DIV2K directory paths."""

    train_hr: Path
    train_lr: Path
    valid_hr: Path
    valid_lr: Path
