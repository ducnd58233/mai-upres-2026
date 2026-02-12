import logging
import os

import torch
import torch.nn as nn
from models import SimpleCNN

TEST_DIR = "runs/models/tests"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def smoke_test(model: nn.Module) -> None:
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == torch.Size([1, 10])
    logger.info(f"\tOutput shape : {out.shape}")
    logger.info(f"\tOutput sample: {out[0, :4].tolist()}")


if __name__ == "__main__":
    os.makedirs(TEST_DIR, exist_ok=True)
    model = SimpleCNN()
    model.eval()

    logger.info("Smoke test...")
    smoke_test(model)

    pt_path = os.path.join(TEST_DIR, "model.pt")
    full_path = os.path.join(TEST_DIR, "model_full.pth")

    torch.save(model, pt_path)
    logger.info(f"Saved nn.Module model -> {pt_path}")

    torch.save(model, full_path)
    logger.info(f"Saved nn.Module model -> {full_path}")
