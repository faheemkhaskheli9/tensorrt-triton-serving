"""Command-line entrypoint for the conversion pipeline.

    python -m trt_serving export --output models/tiny_classifier.onnx
    python -m trt_serving export --output m.onnx --opset 14 --input-shape 1x3x64x64
    python -m trt_serving export --output m.onnx --static-batch
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .export_onnx import ExportConfig, ExportError, export_to_onnx
from .models import TinyClassifier


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

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
