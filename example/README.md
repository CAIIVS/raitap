# raitap example consumer

Standalone project demonstrating how to consume raitap as a third-party library.

## Run

```bash
cd example
uv sync                                                          # pulls bare raitap
uv run raitap --config-name assessment --allow-project-edit      # first run: auto-deps adds extras (or -y for short)
uv run raitap --config-name assessment                           # subsequent runs
```

Or run the equivalent Python entrypoint — it triggers the same auto-deps
flow by passing `auto_install_deps=True` to `raitap.run`:

```bash
uv run python assessment.py
```

The script calls `run(cfg, auto_install_deps=True)`. First run: walks the config,
infers extras, `uv add`s them into this `pyproject.toml`, re-execs.
Subsequent runs short-circuit the bootstrap and proceed straight to the
pipeline.

First-run behaviour: raitap walks `assessment.yaml`, infers which extras the
declared targets need (`captum`, `html`, `metrics`, `torchattacks`, plus a
torch backend like `torch-intel`/`torch-cuda`/`torch-cpu`), and runs
`uv add raitap[<extras>]` to pin them into this `pyproject.toml`. After that
the dependency line evolves automatically as you change the config.

## Layout

- `pyproject.toml` — declares `raitap` only (no extras yet); `tool.uv.package
  = false` so uv manages deps without trying to install this dir as a package.
  Auto-deps fills the extras on first run.
- `assessment.yaml` — composed from bundled raitap groups (`reporting=html`,
  `metrics=classification`) plus inline `model`, `data`, `transparency`,
  `robustness`. Inherits `raitap_schema` for dataclass defaults.
- `assessment.py` — programmatic equivalent of `assessment.yaml` using
  `raitap.AppConfig` + `raitap.run(..., auto_install_deps=True)`. Same pipeline,
  no Hydra CLI. Adapter modules use lazy imports of their wrapped libraries,
  so this file can be imported and run in a venv that has only the base
  `raitap` dep — `auto_install_deps=True` walks the cfg, pins the extras, and
  re-execs.
- Reports land under `outputs/<date>/<time>/`.

## Notes

- During development this project consumes raitap from the parent checkout via
  `[tool.uv.sources]`. Real consumers should remove that section to fetch raitap
  from PyPI.
- Skip auto-deps entirely with `--custom-deps` if you prefer to manage the
  dependency line by hand. See raitap's installation docs.
