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
