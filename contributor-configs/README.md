# Contributor configs

End-to-end use-case configs contributed alongside research / thesis work.
Each subdirectory bundles a self-contained Hydra config (`assessment.yaml`)
plus the data-prep + run scripts needed to reproduce a single demo. They
are **not** part of the public API; treat them as worked examples.

## Layout

```
contributor-configs/
  <use-case>/
    assessment.yaml      # the Hydra config raitap composes
    prep.py | …          # optional data-prep / model-export script
    run.sh               # POSIX entry point
    run.ps1              # Windows PowerShell entry point
    README.md            # use-case-specific steps
    artifacts/           # gitignored — model weights, datasets, etc.
```

Anything under `<use-case>/artifacts/` is gitignored: model weights and
image subsets stay local. The text files (config + scripts + README) are
tracked.

## Running a use case

The `run.sh` / `run.ps1` scripts wrap `raitap` with the right extras and
point Hydra at the bundled config. Two equivalent invocations:

```bash
# POSIX
contributor-configs/<use-case>/run.sh
```

```powershell
# Windows
contributor-configs\<use-case>\run.ps1
```

Both forward extra CLI args to `raitap`, so you can override on the fly:

```bash
contributor-configs/<use-case>/run.sh hardware=cpu data.forward_batch_size=8
```

If you prefer to invoke `raitap` yourself, point Hydra at the use-case
directory with `--config-dir` and pick the config by name:

```bash
uv run raitap \
  --config-dir contributor-configs/<use-case> \
  --config-name assessment
```

## Use cases

- [`lwise-ham10000`](lwise-ham10000/README.md) — dermoscopy classifier
  (L-WISE HAM10000 ResNet-50) with light Captum + Torchattacks coverage.
- [`marabou-mnist`](marabou-mnist/README.md) — tiny MNIST MLP exported to
  ONNX and run through Marabou for formal verification.
