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


def test_cli_build_repo_from_exported_onnx(tmp_path, capsys):
    onnx_path = tmp_path / "m.onnx"
    assert cli.main(["export", "--output", str(onnx_path), "--input-shape", "1x3x16x16"]) == 0

    repo_dir = tmp_path / "repo"
    rc = cli.main(
        ["build-repo", "--model", str(onnx_path), "--repo-dir", str(repo_dir), "--model-name", "m"]
    )

    assert rc == 0
    assert "repository ready" in capsys.readouterr().out
    assert (repo_dir / "m" / "config.pbtxt").is_file()
    assert (repo_dir / "m" / "1" / "model.onnx").is_file()


def test_cli_build_repo_rejects_unsupported_extension(tmp_path, capsys):
    bogus = tmp_path / "m.pt"
    bogus.write_bytes(b"x")
    rc = cli.main(
        ["build-repo", "--model", str(bogus), "--repo-dir", str(tmp_path / "repo"), "--model-name", "m"]
    )
    assert rc == 2
    assert "error" in capsys.readouterr().out
