#!/usr/bin/env bash
# Thesis demo: layer-wise HAM10000 assessment example.
#
# Adapters referenced by examples/lwise-ham10000/assessment.yaml:
#   - CaptumExplainer (LayerGradCam, Saliency, Occlusion, IntegratedGradients)
#   - TorchattacksAssessor (FGSM, PGD)
#   - MarabouAssessor (linf-box)
#   - ClassificationMetrics + HTMLReporter
set -euo pipefail

uv run -p 3.11 \
  --extra torch-cpu \
  --extra captum \
  --extra torchattacks \
  --extra marabou \
  --extra onnx-cpu \
  --extra metrics \
  --extra reporting \
  raitap \
    --config-dir "$PWD/examples/lwise-ham10000" \
    --config-name assessment \
    "$@"
