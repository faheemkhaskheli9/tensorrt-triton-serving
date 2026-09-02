"""PyTorch -> ONNX export with graph validation and numerical parity checks.

The exporter is deliberately strict: it only reports success after the
written ``.onnx`` passes ``onnx.checker`` *and* onnxruntime reproduces the
source model's outputs within tolerance on several random inputs. A failure
raises rather than leaving a plausible-looking but wrong artifact behind,
and the file is written atomically so an interrupted export never leaves a
half-written ``.onnx`` in place.
"""

from __future__ import annotations

import inspect
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

_EXPORT_PARAMS = set(inspect.signature(torch.onnx.export).parameters)

__all__ = [
    "ExportConfig",
    "ExportError",
    "ExportResult",
    "ParityError",
    "export_to_onnx",
]


class ExportError(RuntimeError):
    """The ONNX graph could not be produced or failed validation."""


class ParityError(ExportError):
    """The exported model's outputs diverged from the source PyTorch model."""


@dataclass
class ExportConfig:
    opset: int = 17
    input_shape: tuple[int, ...] = (1, 3, 32, 32)
    input_names: list[str] = field(default_factory=lambda: ["input"])
    output_names: list[str] = field(default_factory=lambda: ["output"])
    dynamic_batch: bool = True
    parity_samples: int = 5
    rtol: float = 1e-3
    atol: float = 1e-5
    seed: int = 0

    def __post_init__(self) -> None:
        self.input_shape = tuple(int(d) for d in self.input_shape)
        if len(self.input_shape) < 2 or any(d <= 0 for d in self.input_shape):
            raise ExportError(f"input_shape must be positive dims (got {self.input_shape})")
        if self.opset < 7:
            raise ExportError(f"opset {self.opset} is too old; use >= 7")


@dataclass
class ExportResult:
    path: Path
    opset: int
    input_shape: tuple[int, ...]
    max_abs_diff: float
    parity_samples: int
    onnx_ir_version: int
    size_bytes: int

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.path} | opset={self.opset} | input={self.input_shape} | "
            f"max_abs_diff={self.max_abs_diff:.2e} over {self.parity_samples} samples | "
            f"{self.size_bytes / 1024:.1f} KiB"
        )


def _dynamic_axes(config: ExportConfig) -> dict[str, dict[int, str]] | None:
    if not config.dynamic_batch:
        return None
    return {name: {0: "batch"} for name in (*config.input_names, *config.output_names)}


def _torch_reference(
    model: torch.nn.Module, inputs: np.ndarray
) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(inputs)).cpu().numpy()


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str | os.PathLike[str],
    config: ExportConfig | None = None,
) -> ExportResult:
    """Export ``model`` to ONNX at ``output_path`` and verify it.

    Raises :class:`ExportError` if the graph is invalid and
    :class:`ParityError` if onnxruntime output drifts past tolerance.
    """
    config = config or ExportConfig()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval()
    torch.manual_seed(config.seed)
    dummy = torch.randn(*config.input_shape)

    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        export_kwargs = dict(
            opset_version=config.opset,
            input_names=list(config.input_names),
            output_names=list(config.output_names),
            dynamic_axes=_dynamic_axes(config),
        )
        # Use the stable TorchScript exporter: it honours `dynamic_axes` /
        # `output_names` predictably and needs no extra packages. Migrating
        # to the torch.export-based exporter is tracked as future work.
        if "dynamo" in _EXPORT_PARAMS:
            export_kwargs["dynamo"] = False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                torch.onnx.export(model, dummy, str(tmp), **export_kwargs)
        except Exception as exc:  # torch raises many concrete types
            raise ExportError(f"torch.onnx.export failed: {exc}") from exc

        try:
            onnx_model = onnx.load(str(tmp))
            onnx.checker.check_model(onnx_model)
        except Exception as exc:
            raise ExportError(f"exported graph failed onnx.checker: {exc}") from exc

        max_abs_diff = _check_parity(model, str(tmp), config)

        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise

    return ExportResult(
        path=target,
        opset=config.opset,
        input_shape=config.input_shape,
        max_abs_diff=max_abs_diff,
        parity_samples=config.parity_samples,
        onnx_ir_version=onnx_model.ir_version,
        size_bytes=target.stat().st_size,
    )


def _check_parity(model: torch.nn.Module, onnx_path: str, config: ExportConfig) -> float:
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    rng = np.random.default_rng(config.seed)

    max_abs_diff = 0.0
    for _ in range(max(config.parity_samples, 1)):
        sample = rng.standard_normal(size=config.input_shape).astype(np.float32)
        expected = _torch_reference(model, sample)
        (actual,) = session.run(None, {input_name: sample})
        diff = float(np.max(np.abs(expected - actual)))
        max_abs_diff = max(max_abs_diff, diff)
        if not np.allclose(expected, actual, rtol=config.rtol, atol=config.atol):
            raise ParityError(
                f"ONNX output diverged from PyTorch: max|Δ|={diff:.3e} "
                f"exceeds rtol={config.rtol}, atol={config.atol}"
            )
    return max_abs_diff
