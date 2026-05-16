# Marabou MNIST

Tiny MNIST MLP exported to ONNX and run through Marabou for formal
verification.

## Requirements

Marabou wheels are Python 3.11 + Linux/WSL only. On Windows, run via WSL.

## Steps

1. **Run the demo.** From the repo root:

   ```bash
   contributor-configs/marabou-mnist/run.sh            # POSIX / WSL
   contributor-configs\marabou-mnist\run.ps1           # Windows (delegates to WSL)
   ```

   The script does three things in order:

   1. `uv sync -p 3.11` with the required extras.
   2. `prep.py` — trains the tiny MLP and exports it to ONNX under
      `artifacts/` (gitignored).
   3. `raitap --config-dir … --config-name assessment` against the bundled
      Hydra config.

   Extra CLI args are forwarded to `raitap`:

   ```bash
   contributor-configs/marabou-mnist/run.sh hardware=cpu
   ```

2. **Inspect outputs.** Marabou verification results land under the Hydra
   run directory (`outputs/<date>/<time>/robustness/`).
