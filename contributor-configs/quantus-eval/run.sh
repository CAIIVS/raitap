#!/usr/bin/env bash
# Quantus explanation-quality grading demo entry point (POSIX).
# Syncs the required extras and runs raitap against the bundled assessment.yaml.
# Forwards extra CLI args.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Run from the repo root: raitap's dev-install detection checks the cwd's
# pyproject.toml, and only then uses `uv sync` (a bare `uv add raitap[...]`
# inside the repo fails as a self-dependency).
cd "$SCRIPT_DIR/../.."

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
