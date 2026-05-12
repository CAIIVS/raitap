#!/usr/bin/env bash
# Thesis demo: layer-wise HAM10000 assessment example.
set -euo pipefail

uv run raitap \
  --config-dir "$PWD/examples/lwise-ham10000" \
  --config-name assessment \
  "$@"
