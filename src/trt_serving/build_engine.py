"""ONNX -> TensorRT engine conversion (Phase 1).

TensorRT only installs on a host with a matching NVIDIA GPU + CUDA driver.
This repository's dev/CI convention is CPU-only, so ``tensorrt`` is never
imported at module load time -- only lazily, inside the real backend -- and
a missing package or unusable GPU raises :class:`TensorRTUnavailableError`
with an actionable message instead of a bare ``ImportError``/CUDA crash.

The actual builder calls are behind :class:`EngineBuilderBackend` /
``backend_factory`` so this module is unit-testable on a CPU-only machine: a
fake backend exercises the config plumbing (precision mode, atomic write,
validation) without ever touching real TensorRT. Swap in the real backend
(the default) on a GPU host to produce a real ``.engine``/``.plan`` file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

__all__ = [
    "EngineBuildResult",
    "EngineBuilderBackend",
    "TensorRTUnavailableError",
    "build_engine",
]

_VALID_PRECISIONS = ("fp32", "fp16")


class TensorRTUnavailableError(RuntimeError):
    """Raised when a TensorRT engine build is requested but impossible here.

    Covers: the ``tensorrt`` package not installed, and no CUDA-capable GPU
    visible to it -- both expected on this repo's CPU-only dev/CI hosts.
    """


@dataclass(frozen=True)
class EngineBuildResult:
    path: Path
    precision: str
    max_workspace_bytes: int
    size_bytes: int


class EngineBuilderBackend(Protocol):
    """Anything that can turn an ONNX file into serialized TensorRT engine bytes."""

    def build(
        self, onnx_path: str, *, precision: str, max_workspace_bytes: int
    ) -> bytes: ...


class _RealTensorRTBackend:
    """The production backend: real TensorRT, only importable on a GPU host."""

    def build(self, onnx_path: str, *, precision: str, max_workspace_bytes: int) -> bytes:
        try:
            import tensorrt as trt  # noqa: PLC0415 - only present on a CUDA host
        except ImportError as exc:
            raise TensorRTUnavailableError(
                "The 'tensorrt' package is not installed. TensorRT requires a "
                "CUDA-capable NVIDIA GPU and matching driver; this repo's CPU-only "
                "dev/CI hosts cannot build a real engine here. Run this on a GPU "
                "host with TensorRT installed, or use --precision/--dry-run to "
                "exercise the rest of the pipeline."
            ) from exc

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, "rb") as handle:
            if not parser.parse(handle.read()):
                errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                raise TensorRTUnavailableError(f"Failed to parse ONNX model: {errors}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_bytes)
        if precision == "fp16":
            if not builder.platform_has_fast_fp16:
                raise TensorRTUnavailableError(
                    "This GPU/build of TensorRT has no fast FP16 support; use precision='fp32'."
                )
            config.set_flag(trt.BuilderFlag.FP16)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise TensorRTUnavailableError("TensorRT engine build returned no serialized network.")
        return bytes(serialized)


def _default_backend_factory() -> EngineBuilderBackend:
    return _RealTensorRTBackend()


def build_engine(
    onnx_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    precision: str = "fp32",
    max_workspace_bytes: int = 1 << 30,
    backend_factory: Callable[[], EngineBuilderBackend] = _default_backend_factory,
) -> EngineBuildResult:
    """Convert ``onnx_path`` into a serialized TensorRT engine at ``output_path``.

    ``output_path`` should end in ``.engine`` or ``.plan``. The file is
    written to a temp path in the same directory and ``os.replace()``d onto
    the target, so a build that fails partway (or raises
    :class:`TensorRTUnavailableError`) never leaves a truncated engine file.
    """
    onnx_source = Path(onnx_path)
    if not onnx_source.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_source}")
    if precision not in _VALID_PRECISIONS:
        raise ValueError(f"precision must be one of {_VALID_PRECISIONS}, got {precision!r}")
    if max_workspace_bytes <= 0:
        raise ValueError(f"max_workspace_bytes must be > 0, got {max_workspace_bytes}")

    target = Path(output_path)
    if target.suffix not in (".engine", ".plan"):
        raise ValueError(f"output_path should end in .engine or .plan, got {target.suffix!r}")
    target.parent.mkdir(parents=True, exist_ok=True)

    backend = backend_factory()
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        engine_bytes = backend.build(
            str(onnx_source), precision=precision, max_workspace_bytes=max_workspace_bytes
        )
        tmp.write_bytes(engine_bytes)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    return EngineBuildResult(
        path=target,
        precision=precision,
        max_workspace_bytes=max_workspace_bytes,
        size_bytes=target.stat().st_size,
    )
