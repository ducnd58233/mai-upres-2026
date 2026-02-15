from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, TypedDict

import torch

logger = logging.getLogger(__name__)


class LoaderEntry(TypedDict):
    loader: Callable[..., tuple[torch.nn.Module, list[int]]]
    required_args: list[str]
    help_extra: str


LOADER_REGISTRY: dict[str, LoaderEntry] = {}


def register_export_loader(
    name: str,
    loader: Callable[..., tuple[torch.nn.Module, list[int]]],
    required_args: list[str],
    help_extra: str = "",
) -> None:
    """Register a model loader for TFLite export."""
    LOADER_REGISTRY[name] = LoaderEntry(
        loader=loader,
        required_args=required_args,
        help_extra=help_extra,
    )


def get_registered_model_types() -> list[str]:
    return sorted(LOADER_REGISTRY.keys())


def get_required_args(model_type: str) -> list[str]:
    """Return required CLI arg names for the given model type."""
    if model_type not in LOADER_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Registered: {get_registered_model_types()}"
        )
    return list(LOADER_REGISTRY[model_type]["required_args"])


def load_for_export(
    model_type: str,
    model_path: str,
    **kwargs: object,
) -> tuple[torch.nn.Module, list[int]]:
    """
    Load a model for export using the registered loader for model_type.

    Args:
        model_type: Registered name (e.g. 'raw', 'antsr').
        model_path: Path to checkpoint.
        **kwargs: Passed to the loader (e.g. height, width for antsr; input_shape for raw).

    Returns:
        (model in eval mode, input_shape for ONNX).
    """
    if model_type not in LOADER_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Registered: {get_registered_model_types()}"
        )
    entry = LOADER_REGISTRY[model_type]
    loader = entry["loader"]
    required = entry["required_args"]
    missing = [k for k in required if k not in kwargs or kwargs[k] is None]
    if missing:
        raise ValueError(
            f"Model type '{model_type}' requires: {required}. Missing: {missing}"
        )
    filtered = {k: kwargs[k] for k in required if k in kwargs}
    return loader(model_path, **filtered)


def _load_raw_checkpoint(
    model_path: str,
    *,
    input_shape: list[int],
) -> tuple[torch.nn.Module, list[int]]:
    """Load a raw PyTorch nn.Module checkpoint (torch.save(model))."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(obj, torch.nn.Module):
        raise ValueError(f"Raw checkpoint must be an nn.Module. Got type: {type(obj)}")
    model = obj
    model.eval()
    for module in model.modules():
        module.training = False

    logger.info("Loaded raw PyTorch nn.Module checkpoint; input_shape=%s", input_shape)
    return model, input_shape


def _register_builtin_loaders() -> None:
    """Register built-in model loaders (raw + antsr)."""
    register_export_loader(
        "raw",
        _load_raw_checkpoint,
        required_args=["input_shape"],
        help_extra="Requires --input-shape (e.g. 1 3 224 224).",
    )
    from models.antsr.export import load_for_export as antsr_load_for_export

    register_export_loader(
        "antsr",
        antsr_load_for_export,
        required_args=["height", "width"],
        help_extra="Requires --height and --width (LR input size).",
    )


_register_builtin_loaders()
