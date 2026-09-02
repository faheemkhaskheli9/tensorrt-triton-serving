"""Small, public sample models used to exercise the conversion pipeline.

Nothing domain-specific -- just a compact CNN so the ONNX / TensorRT stages
have a realistic graph (conv, pooling, linear) without needing a GPU or a
real dataset.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["TinyClassifier"]


class TinyClassifier(nn.Module):
    """A ~40k-param CNN classifier for NCHW image tensors."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
