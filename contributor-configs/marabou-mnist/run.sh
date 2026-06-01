#!/usr/bin/env bash
# Thesis demo: Marabou UC1 — train tiny MNIST MLP, export ONNX, run formal
# verification. Marabou wheels require Python 3.11 on Linux/WSL.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
  python "$SCRIPT_DIR/prep.py"

uv run -p 3.11 \
  --extra torch-cpu \
  --extra marabou \
  --extra onnx-cpu \
  --extra metrics \
  --extra reporting \
  raitap \
    --config-dir "$SCRIPT_DIR" \
    --config-name assessment \
    --custom-deps \
    "$@"
