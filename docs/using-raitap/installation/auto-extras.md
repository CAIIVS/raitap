# Auto-detecting uv extras (`raitap-deps`)

`raitap-deps` reads a Hydra config, infers the required uv extras (backend,
adapter libraries, reporting, tracking), and runs the matching `uv sync` or
`uv add` command. It removes the need to hand-maintain extras lists in
shell scripts and READMEs.

## Quick start

From the repo root:

```bash
uv run raitap-deps --config-dir "$PWD/examples/lwise-ham10000" \
                   --config-name assessment
```

By default it prints a summary frame and executes the resolved command.
Pass `--dry-run` to print only.

## What it infers

- **Backend extra** — `torch-<hw>` or `onnx-<hw>` from the `model.source`
  extension and the resolved hardware suffix.
- **Hardware suffix** — probed from the host (`nvidia-smi` ⇒ `cuda`,
  Intel GPU via `lspci` or `Get-CimInstance` ⇒ `xpu`, otherwise `cpu`;
  macOS is always `cpu`). Override with `--hardware cpu|cuda|xpu`.
- **Adapter extras** — from each top-level `_target_` value (Captum, SHAP,
  Torchattacks, Foolbox, Marabou, MLflow, HTMLReporter→`jinja`,
  PDFReporter→`borb`, metrics).
- **Launcher** — adds `launcher` when `hydra.launcher` resolves into
  `hydra_submitit_launcher`.

## Install modes

`--mode auto` (default) selects `sync` for a developer checkout
(detected via `raitap.utils.diagnostics.is_dev_install`) and `add`
otherwise. Force with `--mode sync` or `--mode add`.

## Safety

Before running uv, the tool validates the inferred set against
`[tool.uv].conflicts` in `pyproject.toml`. A violation prints the
conflicting extras together with the config signals that produced them
and exits with a non-zero status.

## Examples

Inspect what would be installed for a given Hydra preset:

```bash
uv run raitap-deps --dry-run \
    data=mnist_samples model=mlp_mnist robustness=marabou_linf '~transparency'
```

Force CPU even on a CUDA-capable host (useful in CI):

```bash
uv run raitap-deps --hardware cpu --dry-run
```
