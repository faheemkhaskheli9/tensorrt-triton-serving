"""Tests for the Triton model repository generation stage (issue #3).

Building and starting a real `tritonserver` needs the actual Triton binary
(and, for TensorRT, a GPU) that this repo's CPU-only dev/CI host doesn't
have, so these tests instead verify the generated `config.pbtxt` and
directory layout are correct against Triton's documented schema and against
the ONNX graph's own metadata -- the part of the acceptance criteria this
host can actually check.
"""

from __future__ import annotations

import pytest

from trt_serving.export_onnx import ExportConfig, export_to_onnx
from trt_serving.models import TinyClassifier
from trt_serving.repository import (
    ModelRepositoryError,
    RepositoryResult,
    TensorSpec,
    build_model_repository,
)


@pytest.fixture
def onnx_model_dynamic_batch(tmp_path):
    """A real, valid ONNX export with a dynamic batch axis (the CLI default)."""
    model = TinyClassifier(in_channels=3, num_classes=5)
    config = ExportConfig(input_shape=(1, 3, 16, 16), dynamic_batch=True, parity_samples=1)
    out = tmp_path / "model.onnx"
    export_to_onnx(model, out, config)
    return out


@pytest.fixture
def onnx_model_static_batch(tmp_path):
    model = TinyClassifier(in_channels=3, num_classes=5)
    config = ExportConfig(input_shape=(1, 3, 16, 16), dynamic_batch=False, parity_samples=1)
    out = tmp_path / "model_static.onnx"
    export_to_onnx(model, out, config)
    return out


def test_build_repository_from_onnx_dynamic_batch(onnx_model_dynamic_batch, tmp_path):
    result = build_model_repository(
        onnx_model_dynamic_batch, tmp_path / "repo", "tiny_classifier", max_batch_size=8
    )

    assert isinstance(result, RepositoryResult)
    assert result.platform == "onnxruntime_onnx"
    assert result.max_batch_size == 8  # dynamic batch -> caller's value is honoured
    assert result.model_path == tmp_path / "repo" / "tiny_classifier" / "1" / "model.onnx"
    assert result.model_path.exists()
    assert result.config_path == tmp_path / "repo" / "tiny_classifier" / "config.pbtxt"

    text = result.config_path.read_text(encoding="utf-8")
    assert 'name: "tiny_classifier"' in text
    assert 'platform: "onnxruntime_onnx"' in text
    assert "max_batch_size: 8" in text
    assert 'name: "input"' in text
    assert "data_type: TYPE_FP32" in text
    # batch axis excluded from dims -- only the real (C, H, W) dims appear
    assert "dims: [ 3, 16, 16 ]" in text
    assert 'name: "output"' in text
    assert "dims: [ 5 ]" in text


def test_build_repository_from_onnx_static_batch_forces_zero_max_batch(
    onnx_model_static_batch, tmp_path
):
    result = build_model_repository(
        onnx_model_static_batch, tmp_path / "repo", "static_model", max_batch_size=8
    )

    # Triton convention: max_batch_size=0 means the model's own shapes
    # already include the batch dim -- honoured even though the caller
    # asked for 8, per the documented behaviour.
    assert result.max_batch_size == 0
    text = result.config_path.read_text(encoding="utf-8")
    assert "max_batch_size: 0" in text
    # the static batch dim (1) is kept as part of the tensor's real dims
    assert "dims: [ 1, 3, 16, 16 ]" in text


def test_repository_layout_matches_triton_convention(onnx_model_dynamic_batch, tmp_path):
    repo_dir = tmp_path / "repo"
    build_model_repository(onnx_model_dynamic_batch, repo_dir, "m", version=1)

    assert (repo_dir / "m" / "config.pbtxt").is_file()
    assert (repo_dir / "m" / "1" / "model.onnx").is_file()


def test_custom_version_number(onnx_model_dynamic_batch, tmp_path):
    result = build_model_repository(
        onnx_model_dynamic_batch, tmp_path / "repo", "m", version=3
    )
    assert result.model_path == tmp_path / "repo" / "m" / "3" / "model.onnx"


def test_tensorrt_engine_requires_explicit_specs(tmp_path):
    engine = tmp_path / "model.plan"
    engine.write_bytes(b"not a real engine, just needs to exist")

    with pytest.raises(ModelRepositoryError, match="explicit"):
        build_model_repository(engine, tmp_path / "repo", "trt_model")


def test_tensorrt_engine_with_explicit_specs_succeeds(tmp_path):
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"FAKE-ENGINE-BYTES")

    result = build_model_repository(
        engine,
        tmp_path / "repo",
        "trt_model",
        max_batch_size=4,
        inputs=[TensorSpec(name="input", dtype="TYPE_FP32", dims=(3, 16, 16))],
        outputs=[TensorSpec(name="output", dtype="TYPE_FP32", dims=(5,))],
    )

    assert result.platform == "tensorrt_plan"
    assert result.model_path.name == "model.plan"
    assert result.model_path.read_bytes() == b"FAKE-ENGINE-BYTES"


def test_missing_model_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_model_repository(tmp_path / "nope.onnx", tmp_path / "repo", "m")


def test_unsupported_extension_raises(tmp_path):
    bogus = tmp_path / "model.pt"
    bogus.write_bytes(b"x")
    with pytest.raises(ModelRepositoryError, match="unsupported"):
        build_model_repository(bogus, tmp_path / "repo", "m")


def test_invalid_model_name_rejected(onnx_model_dynamic_batch, tmp_path):
    with pytest.raises(ModelRepositoryError, match="invalid model_name"):
        build_model_repository(onnx_model_dynamic_batch, tmp_path / "repo", "bad/name")


def test_invalid_version_rejected(onnx_model_dynamic_batch, tmp_path):
    with pytest.raises(ModelRepositoryError, match="version"):
        build_model_repository(onnx_model_dynamic_batch, tmp_path / "repo", "m", version=0)


def test_rebuild_overwrites_cleanly(onnx_model_dynamic_batch, onnx_model_static_batch, tmp_path):
    repo_dir = tmp_path / "repo"
    build_model_repository(onnx_model_dynamic_batch, repo_dir, "m", max_batch_size=8)
    result = build_model_repository(onnx_model_static_batch, repo_dir, "m", max_batch_size=8)

    # second build's config reflects the second model, not a stale merge of both
    text = result.config_path.read_text(encoding="utf-8")
    assert "max_batch_size: 0" in text
    assert not list((repo_dir / "m").glob("**/.*.tmp"))
