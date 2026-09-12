"""Tests for the ONNX -> TensorRT engine conversion stage (issue #2).

Real TensorRT needs a CUDA GPU that CI/this dev box does not have, so most
tests inject a fake ``backend_factory`` to exercise the config plumbing
(precision, workspace size, atomic write, validation). One test instead
confirms the *real* backend's documented fallback: on a machine without the
`tensorrt` package, calling it raises `TensorRTUnavailableError` with a clear
message rather than a bare ImportError or CUDA crash.
"""

from __future__ import annotations

import pytest

from trt_serving.build_engine import (
    EngineBuildResult,
    TensorRTUnavailableError,
    build_engine,
)


class _FakeBackend:
    """Records what it was called with and returns deterministic bytes."""

    def __init__(self):
        self.calls: list[dict] = []

    def build(self, onnx_path: str, *, precision: str, max_workspace_bytes: int) -> bytes:
        self.calls.append(
            {"onnx_path": onnx_path, "precision": precision, "max_workspace_bytes": max_workspace_bytes}
        )
        return f"FAKE-ENGINE:{precision}:{max_workspace_bytes}".encode()


class _BoomBackend:
    def build(self, onnx_path: str, *, precision: str, max_workspace_bytes: int) -> bytes:
        raise TensorRTUnavailableError("synthetic build failure")


@pytest.fixture
def onnx_file(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"not a real onnx graph, just needs to exist")
    return path


def test_build_engine_produces_engine_artifact(onnx_file, tmp_path):
    out = tmp_path / "model.engine"
    backend = _FakeBackend()

    result = build_engine(onnx_file, out, backend_factory=lambda: backend)

    assert isinstance(result, EngineBuildResult)
    assert out.exists()
    assert out.read_bytes() == b"FAKE-ENGINE:fp32:1073741824"
    assert result.size_bytes == out.stat().st_size


def test_precision_mode_is_configurable_and_plumbed_through(onnx_file, tmp_path):
    out = tmp_path / "model.plan"
    backend = _FakeBackend()

    result = build_engine(onnx_file, out, precision="fp16", backend_factory=lambda: backend)

    assert result.precision == "fp16"
    assert backend.calls[0]["precision"] == "fp16"


def test_invalid_precision_rejected(onnx_file, tmp_path):
    with pytest.raises(ValueError, match="precision"):
        build_engine(onnx_file, tmp_path / "m.engine", precision="int8-please")


def test_max_workspace_bytes_is_configurable(onnx_file, tmp_path):
    backend = _FakeBackend()
    build_engine(
        onnx_file,
        tmp_path / "m.engine",
        max_workspace_bytes=64 * (1 << 20),
        backend_factory=lambda: backend,
    )
    assert backend.calls[0]["max_workspace_bytes"] == 64 * (1 << 20)


def test_invalid_max_workspace_bytes_rejected(onnx_file, tmp_path):
    with pytest.raises(ValueError, match="max_workspace_bytes"):
        build_engine(onnx_file, tmp_path / "m.engine", max_workspace_bytes=0)


def test_missing_onnx_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_engine(tmp_path / "nope.onnx", tmp_path / "m.engine")


def test_output_extension_is_validated(onnx_file, tmp_path):
    with pytest.raises(ValueError, match=r"\.engine or \.plan"):
        build_engine(onnx_file, tmp_path / "m.onnx", backend_factory=lambda: _FakeBackend())


def test_build_failure_leaves_no_partial_artifact(onnx_file, tmp_path):
    out = tmp_path / "m.engine"
    with pytest.raises(TensorRTUnavailableError):
        build_engine(onnx_file, out, backend_factory=lambda: _BoomBackend())

    assert not out.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_real_backend_raises_clear_error_without_tensorrt_installed(onnx_file, tmp_path):
    # This dev/CI host is CPU-only per the repo convention: `tensorrt` is not
    # installed, so the default (real) backend must fail loud and clear
    # rather than a bare ImportError or CUDA crash.
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("a real TensorRT install is present; the fallback path is not exercised")

    out = tmp_path / "m.engine"
    with pytest.raises(TensorRTUnavailableError, match="tensorrt"):
        build_engine(onnx_file, out)

    assert not out.exists()
