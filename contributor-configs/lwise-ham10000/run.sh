#!/usr/bin/env bash
# L-WISE HAM10000 demo entry point. Syncs the required extras and runs
# raitap against the bundled assessment.yaml. Forwards extra CLI args.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv sync \
  --extra torch-cpu \
  --extra captum \
  --extra metrics \
  --extra reporting \
  --extra torchattacks

uv run \
  --extra torch-cpu \
  --extra captum \
  --extra metrics \
  --extra reporting \
  --extra torchattacks \
  raitap \
    --config-dir "$SCRIPT_DIR" \
    --config-name assessment \
    "$@"
