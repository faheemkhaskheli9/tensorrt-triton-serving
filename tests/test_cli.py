"""Tests for the conversion CLI."""

from __future__ import annotations

import onnx
import pytest

from trt_serving import cli


def test_cli_export_default(tmp_path, capsys):
    out = tmp_path / "m.onnx"
    rc = cli.main(["export", "--output", str(out)])
    assert rc == 0
    assert "exported" in capsys.readouterr().out
    onnx.checker.check_model(onnx.load(str(out)))


def test_cli_export_custom_shape_and_opset(tmp_path):
    out = tmp_path / "m.onnx"
    rc = cli.main(
        ["export", "--output", str(out), "--opset", "14", "--input-shape", "1x1x16x16"]
    )
    assert rc == 0
    model = onnx.load(str(out))
    assert {imp.domain: imp.version for imp in model.opset_import}[""] == 14


def test_cli_bad_shape_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["export", "--output", str(tmp_path / "m.onnx"), "--input-shape", "abc"])


def test_cli_requires_subcommand():
    with pytest.raises(SystemExit):
        cli.main([])
