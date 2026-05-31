#!/usr/bin/env bash
# Average-case (ImageNet-C corruptions) demo entry point (POSIX).
# Syncs the required extras and runs raitap against the bundled assessment.yaml.
# Forwards extra CLI args.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXTRAS=(
  --extra torch-cpu
  --extra imagecorruptions
  --extra metrics
  --extra reporting
)

uv sync "${EXTRAS[@]}"
uv run "${EXTRAS[@]}" raitap \
  --config-dir "$SCRIPT_DIR" \
  --config-name assessment \
  "$@"
