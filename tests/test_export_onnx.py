"""Tests for the PyTorch -> ONNX export stage (issue #1)."""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from trt_serving.export_onnx import (
    ExportConfig,
    ExportError,
    ParityError,
    export_to_onnx,
)
from trt_serving.models import TinyClassifier


def test_export_produces_valid_onnx(tmp_path):
    out = tmp_path / "model.onnx"
    result = export_to_onnx(TinyClassifier(), out)

    assert out.exists()
    onnx.checker.check_model(onnx.load(str(out)))
    assert result.opset == 17
    assert result.size_bytes > 0


def test_export_output_matches_pytorch(tmp_path):
    model = TinyClassifier()
    result = export_to_onnx(model, tmp_path / "m.onnx", ExportConfig(parity_samples=8))

    assert result.max_abs_diff < 1e-4

    # independent re-check with a fresh input
    sample = np.random.default_rng(1).standard_normal((1, 3, 32, 32)).astype(np.float32)
    with torch.no_grad():
        expected = model.eval()(torch.from_numpy(sample)).numpy()
    sess = ort.InferenceSession(str(tmp_path / "m.onnx"), providers=["CPUExecutionProvider"])
    (actual,) = sess.run(None, {sess.get_inputs()[0].name: sample})
    np.testing.assert_allclose(expected, actual, rtol=1e-3, atol=1e-5)


def test_opset_is_configurable(tmp_path):
    for opset in (14, 17):
        out = tmp_path / f"op{opset}.onnx"
        export_to_onnx(TinyClassifier(), out, ExportConfig(opset=opset))
        model = onnx.load(str(out))
        default_domain = {imp.domain: imp.version for imp in model.opset_import}
        assert default_domain[""] == opset


def test_input_shape_is_configurable(tmp_path):
    cfg = ExportConfig(input_shape=(2, 1, 16, 16))
    out = tmp_path / "s.onnx"
    export_to_onnx(TinyClassifier(in_channels=1), out, cfg)

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    sample = np.zeros((2, 1, 16, 16), dtype=np.float32)
    (logits,) = sess.run(None, {sess.get_inputs()[0].name: sample})
    assert logits.shape == (2, 10)


def test_dynamic_batch_axis_allows_varied_batch(tmp_path):
    out = tmp_path / "dyn.onnx"
    export_to_onnx(TinyClassifier(), out, ExportConfig(dynamic_batch=True))
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for batch in (1, 4, 8):
        (logits,) = sess.run(None, {name: np.zeros((batch, 3, 32, 32), dtype=np.float32)})
        assert logits.shape == (batch, 10)


def test_static_batch_rejects_other_batch_sizes(tmp_path):
    out = tmp_path / "static.onnx"
    export_to_onnx(TinyClassifier(), out, ExportConfig(dynamic_batch=False))
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    with pytest.raises(Exception):
        sess.run(None, {name: np.zeros((4, 3, 32, 32), dtype=np.float32)})


def test_export_is_atomic_on_checker_failure(tmp_path, monkeypatch):
    out = tmp_path / "bad.onnx"

    def _boom(_model):
        raise onnx.checker.ValidationError("synthetic")

    monkeypatch.setattr("trt_serving.export_onnx.onnx.checker.check_model", _boom)
    with pytest.raises(ExportError):
        export_to_onnx(TinyClassifier(), out)

    assert not out.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_parity_failure_raises_and_leaves_no_file(tmp_path, monkeypatch):
    out = tmp_path / "drift.onnx"
    real_session = ort.InferenceSession

    class _NoisySession:
        def __init__(self, *a, **k):
            self._inner = real_session(*a, **k)

        def get_inputs(self):
            return self._inner.get_inputs()

        def run(self, *a, **k):
            return [o + 5.0 for o in self._inner.run(*a, **k)]

    monkeypatch.setattr("trt_serving.export_onnx.ort.InferenceSession", _NoisySession)
    with pytest.raises(ParityError):
        export_to_onnx(TinyClassifier(), out)
    assert not out.exists()


def test_invalid_config_rejected():
    with pytest.raises(ExportError):
        ExportConfig(input_shape=(1, 3, 0, 32))
    with pytest.raises(ExportError):
        ExportConfig(opset=3)


def test_export_overwrites_existing_file(tmp_path):
    out = tmp_path / "m.onnx"
    out.write_bytes(b"stale")
    export_to_onnx(TinyClassifier(), out)
    onnx.checker.check_model(onnx.load(str(out)))
