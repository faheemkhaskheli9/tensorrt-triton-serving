# Triton / TensorRT Model Serving

> AI Infrastructure / Deployment portfolio project — independent open-source implementation.
> This is an original, from-scratch build. It is not affiliated with, and does not
> contain any code, prompts, data, or business logic from, any employer or client.

![status](https://img.shields.io/badge/status-in%20progress-yellow)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## 1. Problem

Achieving low-latency, high-throughput GPU inference in production often requires converting models through ONNX/TensorRT and serving via Triton.

## 2. Architecture

```text
PyTorch Model -> ONNX -> TensorRT -> Triton Inference Server -> FastAPI Gateway -> Client
```

## 3. Technology Stack

- PyTorch
- ONNX
- TensorRT
- Triton Inference Server
- FastAPI
- Docker

## 4. Feature List

- PyTorch to ONNX conversion
- ONNX to TensorRT optimization
- Triton Inference Server deployment
- FastAPI gateway in front of Triton
- Latency benchmarking
- Throughput benchmarking
- GPU utilization monitoring

## 5. Implementation Plan

1. Phase 1: Model conversion pipeline (PyTorch -> ONNX -> TensorRT)
2. Phase 2: Triton deployment configuration
3. Phase 3: FastAPI gateway and client integration
4. Phase 4: Latency/throughput/GPU utilization benchmarking

## 6. Repository Structure

```text
tensorrt-triton-serving/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── .env.example
├── docker/
├── docs/
│   ├── architecture.md
│   └── evaluation.md
├── src/
├── tests/
├── configs/
├── scripts/
├── notebooks/
├── examples/
├── assets/
└── .github/
    └── workflows/
```

## 7. Setup

```bash
git clone <this-repo-url>
cd tensorrt-triton-serving
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or: pip install -e .
cp .env.example .env              # fill in API keys / config
```

## 8. Dataset

Document which public dataset(s) or synthetic data generators are used here.
No proprietary, employer-owned, or client-identifiable data is used in this project.

## 9. Training / Execution

Phase 1 — PyTorch -> ONNX conversion (implemented, CPU-only):

```bash
pip install -r requirements.txt
export PYTHONPATH=src            # or: pip install -e .

# Export the bundled sample CNN to ONNX (validates the graph and checks
# numerical parity against the PyTorch model before writing the file):
python -m trt_serving export --output models/tiny_classifier.onnx --opset 17

# Configurable input shape / opset / batch axis:
python -m trt_serving export --output models/m.onnx --input-shape 1x3x64x64 --static-batch
```

The exporter writes atomically and raises rather than emit an `.onnx` that
fails `onnx.checker` or drifts from the source model.

Phase 1 — ONNX -> TensorRT engine conversion (needs a CUDA GPU host with
`tensorrt` installed; not covered by the base CPU-only install):

```bash
python -m trt_serving build-engine --onnx models/tiny_classifier.onnx \
    --output models/tiny_classifier.engine --precision fp16
```

On a CPU-only host (no `tensorrt` package / no CUDA GPU) this raises
`TensorRTUnavailableError` with an actionable message instead of a bare
`ImportError` or CUDA crash; the conversion logic itself (precision mode,
workspace size, atomic write) is unit-tested behind a fake builder backend
so it doesn't need a GPU to verify.

Phase 2 — Triton model repository layout:

```bash
python -m trt_serving build-repo --model models/tiny_classifier.onnx \
    --repo-dir models/triton_repo --model-name tiny_classifier
```

Writes `models/triton_repo/tiny_classifier/config.pbtxt` and
`models/triton_repo/tiny_classifier/1/model.onnx`. For an ONNX model, the
`config.pbtxt` input/output section (names, dtypes, dims) is derived from the
graph itself rather than hand-typed. A `.engine`/`.plan` TensorRT file has no
introspectable graph metadata here, so its tensor specs must be supplied via
the `build_model_repository(..., inputs=..., outputs=...)` Python API.
Starting `tritonserver` against the generated repository and confirming the
model reports ready needs the real Triton binary (and, for TensorRT, a GPU),
which this repo's CPU-only dev/CI host does not have.

## 10. Evaluation

Document evaluation metrics and how to reproduce them here (see `docs/evaluation.md`).

## 11. Results

_To be filled in as the implementation progresses — screenshots, metrics tables, and
sample outputs go here._

## 12. API

_If this project exposes an API, document the main endpoints here (or link to
auto-generated OpenAPI docs, e.g. `/docs` for FastAPI)._

## 13. Docker

```bash
docker build -t tensorrt-triton-serving .
docker run -p 8000:8000 tensorrt-triton-serving
```

## 14. Tests

```bash
pytest tests/
```

## 15. Limitations

- This is a from-scratch, independent recreation built for portfolio purposes.
- Performance numbers, once added, are based on public datasets and are not
  representative of any production system's real-world results.

## 16. Future Work

- Expand evaluation coverage and add CI-based regression checks.
- Add more configuration presets and deployment targets.
- Track open items as GitHub Issues.

## 17. Disclosure

This repository is an **independent open-source recreation inspired by the kind of
production systems I have worked on professionally**. It contains no employer or
client source code, prompts, datasets, credentials, architecture diagrams, or
business logic. All code, data, and documentation here are original or built on
publicly available datasets and open-source tools.

---
_Last updated: 2026-09-02_
