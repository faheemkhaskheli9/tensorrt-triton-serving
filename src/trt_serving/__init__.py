"""TensorRT / Triton serving pipeline.

Phase 1 is the model-conversion pipeline (PyTorch -> ONNX -> TensorRT).
This module currently exposes the PyTorch -> ONNX stage: a configurable
exporter that validates the graph and checks numerical parity against the
source model.
"""

from .export_onnx import (
    ExportConfig,
    ExportError,
    ExportResult,
    ParityError,
    export_to_onnx,
)
from .models import TinyClassifier

__all__ = [
    "ExportConfig",
    "ExportError",
    "ExportResult",
    "ParityError",
    "TinyClassifier",
    "export_to_onnx",
]
