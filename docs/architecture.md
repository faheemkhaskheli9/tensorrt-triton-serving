# Architecture Notes: Triton / TensorRT Model Serving

## Pipeline

```text
PyTorch Model -> ONNX -> TensorRT -> Triton Inference Server -> FastAPI Gateway -> Client
```

## Components

- PyTorch to ONNX conversion
- ONNX to TensorRT optimization
- Triton Inference Server deployment
- FastAPI gateway in front of Triton
- Latency benchmarking
- Throughput benchmarking
- GPU utilization monitoring

## Design Notes

- Keep provider/model choices swappable behind interfaces (see `multi-llm-router`
  and similar projects in this portfolio for the general pattern).
- Prefer configuration-driven pipelines (YAML/JSON in `configs/`) over hardcoded
  parameters so experiments are reproducible.
