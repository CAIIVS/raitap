#!/usr/bin/env bash
# Faster R-CNN / Udacity detection demo entry point. Syncs the required
# extras and runs raitap against the bundled assessment.yaml. Forwards
# extra CLI args.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv sync \
  --extra torch-cpu \
  --extra captum \
  --extra metrics \
  --extra reporting

# Materialise sample images at native resolution (idempotent). The demo
# sample loader resizes to 224x224 which is fatal for detection; this step
# copies the cached images into ``images/`` so ``data.source`` (a real path)
# bypasses the demo loader.
uv run python "$SCRIPT_DIR/scripts/fetch_images.py"

# --acknowledge-preprocessing-off silences the no-preprocessing warning;
# torchvision detection models do their own internal resize/normalise, and
# adding a RAITAP-level preprocessing step would break box coord alignment.
uv run \
  --extra torch-cpu \
  --extra captum \
  --extra metrics \
  --extra reporting \
  raitap \
    --config-dir "$SCRIPT_DIR" \
    --config-name assessment \
    --acknowledge-preprocessing-off \
    "$@"
