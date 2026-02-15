import logging
import os

import torch
import torch.nn as nn

from configs import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

TEST_DIR = "runs/models/tests"


class SimpleCNN(nn.Module):
    """Small image classifier: [B, 3, 224, 224] → [B, 10]"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → [B, 16, 112, 112]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → [B, 32,  56,  56]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # → [B, 64,   4,   4]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


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
