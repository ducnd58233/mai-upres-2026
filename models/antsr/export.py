from __future__ import annotations

import logging
from pathlib import Path

import torch

from models.antsr.model import AntSR

logger = logging.getLogger(__name__)


def load_for_export(
    model_path: str,
    *,
    height: int,
    width: int,
) -> tuple[torch.nn.Module, list[int]]:
    """
    Load an AntSR deploy checkpoint for export (ONNX/TFLite).

    Args:
        model_path: Path to .pt checkpoint (dict with 'model' + 'cfg').
        height: Input LR height for ONNX input shape.
        width: Input LR width for ONNX input shape.

    Returns:
        (model in eval mode, input_shape e.g. [1, 3, height, width]).
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt or "cfg" not in ckpt:
        raise ValueError(
            "AntSR checkpoint must be a dict with 'model' and 'cfg'. "
            f"Got keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
        )
    cfg = dict(ckpt["cfg"])
    state = ckpt["model"]

    model = AntSR(deploy=True, **cfg)
    model.load_state_dict(state, strict=True)
    model.eval()
    for module in model.modules():
        module.training = False

    input_shape = [1, 3, height, width]
    logger.info(
        "Loaded AntSR deploy checkpoint for export; input_shape=%s", input_shape
    )
    return model, input_shape
