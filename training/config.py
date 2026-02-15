from __future__ import annotations

RUNS_MODELS_ROOT = "runs/models"


def get_default_out_dir(model_name: str) -> str:
    return f"{RUNS_MODELS_ROOT}/{model_name}"
