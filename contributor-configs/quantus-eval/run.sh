#!/usr/bin/env bash
# Quantus explanation-quality grading demo entry point (POSIX).
# Syncs the required extras and runs raitap against the bundled assessment.yaml.
# Forwards extra CLI args.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXTRAS=(
  --extra torch-cpu
  --extra captum
  --extra quantus
  --extra metrics
  --extra html
)

uv sync "${EXTRAS[@]}"
uv run "${EXTRAS[@]}" raitap \
  --config-dir "$SCRIPT_DIR" \
  --config-name assessment \
  "$@"
