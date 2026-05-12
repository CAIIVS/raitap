#!/usr/bin/env bash
# Thesis demo: Marabou UC1 — train tiny MNIST MLP, export ONNX, run formal
# verification. Marabou wheels require Python 3.11 on Linux/WSL.
set -euo pipefail

uv sync -p 3.11 \
  --extra torch-cpu \
  --extra marabou \
  --extra onnx-cpu \
  --extra metrics \
  --extra reporting

uv run -p 3.11 \
  --extra torch-cpu \
  --extra onnx-cpu \
  --with onnxscript \
  python scripts/prep_uc1_mnist.py

uv run -p 3.11 \
  --extra torch-cpu \
  --extra marabou \
  --extra onnx-cpu \
  --extra metrics \
  --extra reporting \
  raitap data=mnist_samples model=mlp_mnist robustness=marabou_linf "$@"
