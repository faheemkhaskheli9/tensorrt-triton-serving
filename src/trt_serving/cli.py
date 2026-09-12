"""Command-line entrypoint for the conversion pipeline.

    python -m trt_serving export --output models/tiny_classifier.onnx
    python -m trt_serving export --output m.onnx --opset 14 --input-shape 1x3x64x64
    python -m trt_serving export --output m.onnx --static-batch
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .build_engine import TensorRTUnavailableError, build_engine
from .export_onnx import ExportConfig, ExportError, export_to_onnx
from .models import TinyClassifier
from .repository import ModelRepositoryError, build_model_repository


def _parse_shape(text: str) -> tuple[int, ...]:
    try:
        dims = tuple(int(part) for part in text.lower().replace("x", ",").split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad shape {text!r} (use e.g. 1x3x32x32)") from exc
    if not dims:
        raise argparse.ArgumentTypeError("shape is empty")
    return dims


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trt_serving")
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="Export the sample model to ONNX.")
    export.add_argument("--output", required=True, help="Destination .onnx path.")
    export.add_argument("--opset", type=int, default=17)
    export.add_argument(
        "--input-shape", type=_parse_shape, default=(1, 3, 32, 32),
        help="NCHW shape, e.g. 1x3x32x32.",
    )
    export.add_argument("--num-classes", type=int, default=10)
    export.add_argument("--static-batch", action="store_true", help="Disable dynamic batch axis.")
    export.add_argument("--parity-samples", type=int, default=5)

    build = sub.add_parser(
        "build-engine", help="Convert an ONNX model into a TensorRT engine (needs a GPU host)."
    )
    build.add_argument("--onnx", required=True, help="Source .onnx file.")
    build.add_argument("--output", required=True, help="Destination .engine/.plan path.")
    build.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    build.add_argument(
        "--max-workspace-mb", type=int, default=1024, help="TensorRT builder workspace, in MiB."
    )

    repo = sub.add_parser(
        "build-repo", help="Generate a Triton model repository entry for a converted model."
    )
    repo.add_argument("--model", required=True, help="Source .onnx/.plan/.engine file.")
    repo.add_argument("--repo-dir", required=True, help="Triton model repository root.")
    repo.add_argument("--model-name", required=True, help="Name Triton will serve this model as.")
    repo.add_argument("--version", type=int, default=1)
    repo.add_argument(
        "--max-batch-size", type=int, default=8,
        help="Used only if the model's batch axis is dynamic.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "export":
        in_channels = args.input_shape[1] if len(args.input_shape) >= 2 else 3
        model = TinyClassifier(in_channels=in_channels, num_classes=args.num_classes)
        config = ExportConfig(
            opset=args.opset,
            input_shape=args.input_shape,
            dynamic_batch=not args.static_batch,
            parity_samples=args.parity_samples,
        )
        try:
            result = export_to_onnx(model, args.output, config)
        except ExportError as exc:
            print(f"error: {exc}")
            return 2
        print(f"exported {result}")
        return 0

    if args.command == "build-engine":
        try:
            result = build_engine(
                args.onnx,
                args.output,
                precision=args.precision,
                max_workspace_bytes=args.max_workspace_mb * (1 << 20),
            )
        except TensorRTUnavailableError as exc:
            print(f"error: {exc}")
            return 3
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}")
            return 2
        print(
            f"built {result.path} | precision={result.precision} | "
            f"{result.size_bytes / 1024:.1f} KiB"
        )
        return 0

    if args.command == "build-repo":
        try:
            result = build_model_repository(
                args.model,
                args.repo_dir,
                args.model_name,
                version=args.version,
                max_batch_size=args.max_batch_size,
            )
        except (ModelRepositoryError, FileNotFoundError) as exc:
            print(f"error: {exc}")
            return 2
        print(
            f"repository ready at {result.model_dir} | platform={result.platform} | "
            f"max_batch_size={result.max_batch_size}"
        )
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
