"""Triton model repository layout generation (Phase 2).

Given a converted model file (an ONNX graph from :mod:`export_onnx`, or a
TensorRT engine from :mod:`build_engine`), builds the on-disk model
repository layout Triton requires::

    <repo_dir>/<model_name>/
        config.pbtxt
        1/
            model.onnx   (or model.plan for a TensorRT engine)

For an ONNX model, the ``config.pbtxt`` input/output section is derived
directly from the graph's own value-info (names, dtypes, dims) rather than
hand-typed, so the config can't silently drift from the model it describes.
A TensorRT ``.engine``/``.plan`` file carries no such introspectable graph
metadata in this repo's CPU-only toolchain, so its tensor specs must be
supplied explicitly.

Actually starting ``tritonserver`` and confirming the model reports ready
needs the real Triton binary (and, for the TensorRT platform, a GPU) which
this repo's CPU-only dev/CI convention does not have (see README §15); the
acceptance bar here is a structurally correct repository per Triton's
documented layout and config schema, which the tests verify against the
model's own metadata.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import onnx

__all__ = [
    "ModelRepositoryError",
    "TensorSpec",
    "RepositoryResult",
    "build_model_repository",
]

# ONNX TensorProto.DataType -> Triton config.pbtxt data_type. Only the types
# this pipeline's sample models plausibly produce are mapped; anything else
# raises rather than emitting a bogus/guessed data_type.
_ONNX_TO_TRITON_DTYPE = {
    1: "TYPE_FP32",  # FLOAT
    2: "TYPE_UINT8",
    3: "TYPE_INT8",
    5: "TYPE_INT16",
    6: "TYPE_INT32",
    7: "TYPE_INT64",
    9: "TYPE_BOOL",
    10: "TYPE_FP16",
    11: "TYPE_FP64",
}

_PLATFORM_BY_SUFFIX = {
    ".onnx": "onnxruntime_onnx",
    ".plan": "tensorrt_plan",
    ".engine": "tensorrt_plan",
}


class ModelRepositoryError(RuntimeError):
    """The model repository could not be built from the given model file."""


@dataclass(frozen=True)
class TensorSpec:
    """One input/output tensor as it appears in ``config.pbtxt`` (batch dim excluded)."""

    name: str
    dtype: str
    dims: tuple[int, ...]


@dataclass(frozen=True)
class RepositoryResult:
    model_dir: Path
    config_path: Path
    model_path: Path
    platform: str
    max_batch_size: int
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]


def _tensor_specs_from_onnx(
    value_infos: Sequence, initializer_names: set[str]
) -> tuple[list[TensorSpec], bool]:
    """Return (specs, dynamic_batch) for a list of ONNX graph input/output protos."""
    specs: list[TensorSpec] = []
    dynamic_batch_flags: set[bool] = set()

    for vi in value_infos:
        if vi.name in initializer_names:
            continue  # a learned weight, not a real input

        elem_type = vi.type.tensor_type.elem_type
        try:
            dtype = _ONNX_TO_TRITON_DTYPE[elem_type]
        except KeyError as exc:
            raise ModelRepositoryError(
                f"tensor {vi.name!r} has unsupported ONNX elem_type {elem_type}"
            ) from exc

        dims_proto = list(vi.type.tensor_type.shape.dim)
        if not dims_proto:
            raise ModelRepositoryError(f"tensor {vi.name!r} has no shape information")

        is_dynamic_batch = bool(dims_proto[0].dim_param)
        dynamic_batch_flags.add(is_dynamic_batch)
        rest = dims_proto[1:] if is_dynamic_batch else dims_proto

        dims: list[int] = []
        for dim in rest:
            if dim.dim_param:
                raise ModelRepositoryError(
                    f"tensor {vi.name!r} has a dynamic non-batch dimension "
                    f"({dim.dim_param!r}); Triton's config.pbtxt requires fixed dims "
                    "for every axis but the batch axis"
                )
            if dim.dim_value <= 0:
                raise ModelRepositoryError(
                    f"tensor {vi.name!r} has a non-positive dimension ({dim.dim_value})"
                )
            dims.append(dim.dim_value)

        specs.append(TensorSpec(name=vi.name, dtype=dtype, dims=tuple(dims)))

    if len(dynamic_batch_flags) > 1:
        raise ModelRepositoryError(
            "inconsistent batch dynamism across tensors: some have a dynamic "
            "first dim and some don't; export the model with a uniform batch axis"
        )
    dynamic_batch = dynamic_batch_flags.pop() if dynamic_batch_flags else False
    return specs, dynamic_batch


def _introspect_onnx(model_path: Path) -> tuple[list[TensorSpec], list[TensorSpec], bool]:
    try:
        onnx_model = onnx.load(str(model_path))
    except Exception as exc:  # onnx raises assorted concrete types
        raise ModelRepositoryError(f"failed to load ONNX model {model_path}: {exc}") from exc

    initializer_names = {init.name for init in onnx_model.graph.initializer}
    inputs, in_dynamic = _tensor_specs_from_onnx(onnx_model.graph.input, initializer_names)
    outputs, out_dynamic = _tensor_specs_from_onnx(onnx_model.graph.output, set())
    if not inputs:
        raise ModelRepositoryError(f"ONNX model {model_path} has no real (non-weight) inputs")
    if not outputs:
        raise ModelRepositoryError(f"ONNX model {model_path} declares no outputs")
    if in_dynamic != out_dynamic:
        raise ModelRepositoryError(
            "inputs and outputs disagree on whether the batch axis is dynamic"
        )
    return inputs, outputs, in_dynamic


def _render_tensor_block(kind: str, specs: Sequence[TensorSpec]) -> str:
    entries = []
    for spec in specs:
        dims = ", ".join(str(d) for d in spec.dims)
        entries.append(
            "  {\n"
            f'    name: "{spec.name}"\n'
            f"    data_type: {spec.dtype}\n"
            f"    dims: [ {dims} ]\n"
            "  }"
        )
    return f"{kind} [\n" + ",\n".join(entries) + "\n]\n"


def _render_config(
    model_name: str,
    platform: str,
    max_batch_size: int,
    inputs: Sequence[TensorSpec],
    outputs: Sequence[TensorSpec],
) -> str:
    parts = [
        f'name: "{model_name}"\n',
        f'platform: "{platform}"\n',
        f"max_batch_size: {max_batch_size}\n",
        _render_tensor_block("input", inputs),
        _render_tensor_block("output", outputs),
    ]
    return "\n".join(parts)


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_write_text(text: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def build_model_repository(
    model_path: str | os.PathLike[str],
    repo_dir: str | os.PathLike[str],
    model_name: str,
    *,
    version: int = 1,
    max_batch_size: int = 8,
    inputs: Sequence[TensorSpec] | None = None,
    outputs: Sequence[TensorSpec] | None = None,
) -> RepositoryResult:
    """Build a Triton model repository entry for ``model_path`` under ``repo_dir``.

    For an ``.onnx`` model, ``inputs``/``outputs`` are introspected from the
    graph automatically; passing them explicitly is only needed (and required)
    for a ``.engine``/``.plan`` TensorRT file, which carries no portable
    graph metadata this pipeline can read without a GPU/TensorRT install.

    ``max_batch_size`` is honoured only when the model's own batch axis is
    dynamic; a model exported with a static batch dim forces
    ``max_batch_size=0`` (Triton's convention for "the model's shapes already
    include any batching"), regardless of what is passed.

    The config file and the served model file are each written to a temp
    path in their target directory and ``os.replace()``d into place, so an
    interrupted build never leaves a partial ``config.pbtxt`` or model file
    that Triton could load as if it were complete.
    """
    source = Path(model_path)
    if not source.exists():
        raise FileNotFoundError(f"model file not found: {source}")
    if not model_name or "/" in model_name or "\\" in model_name:
        raise ModelRepositoryError(f"invalid model_name: {model_name!r}")
    if version < 1:
        raise ModelRepositoryError(f"version must be >= 1, got {version}")

    try:
        platform = _PLATFORM_BY_SUFFIX[source.suffix]
    except KeyError as exc:
        raise ModelRepositoryError(
            f"unsupported model file extension {source.suffix!r}; "
            f"expected one of {sorted(_PLATFORM_BY_SUFFIX)}"
        ) from exc

    if platform == "onnxruntime_onnx":
        introspected_inputs, introspected_outputs, dynamic_batch = _introspect_onnx(source)
        resolved_inputs = tuple(inputs) if inputs is not None else tuple(introspected_inputs)
        resolved_outputs = tuple(outputs) if outputs is not None else tuple(introspected_outputs)
        resolved_max_batch = max_batch_size if dynamic_batch else 0
        served_name = "model.onnx"
    else:
        if not inputs or not outputs:
            raise ModelRepositoryError(
                "a TensorRT .engine/.plan file has no introspectable graph metadata "
                "here; pass explicit `inputs=` / `outputs=` TensorSpec sequences"
            )
        resolved_inputs = tuple(inputs)
        resolved_outputs = tuple(outputs)
        resolved_max_batch = max_batch_size
        served_name = "model.plan"

    model_dir = Path(repo_dir) / model_name
    version_dir = model_dir / str(version)
    model_target = version_dir / served_name
    config_target = model_dir / "config.pbtxt"

    _atomic_copy(source, model_target)
    config_text = _render_config(
        model_name, platform, resolved_max_batch, resolved_inputs, resolved_outputs
    )
    _atomic_write_text(config_text, config_target)

    return RepositoryResult(
        model_dir=model_dir,
        config_path=config_target,
        model_path=model_target,
        platform=platform,
        max_batch_size=resolved_max_batch,
        inputs=resolved_inputs,
        outputs=resolved_outputs,
    )
