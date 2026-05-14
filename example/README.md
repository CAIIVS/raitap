# raitap example consumer

Standalone project demonstrating how to consume raitap as a third-party library.

## Run

```bash
cd example
uv sync
uv run raitap --config-name assessment
```

## Layout

- `pyproject.toml` — declares raitap as a dependency.
- `configs/assessment.yaml` — composed from bundled raitap groups (`reporting=html`,
  `metrics=classification`) plus inline `model`, `data`, `transparency`, `robustness`.
- Reports land under `outputs/<date>/<time>/`.

## Notes

- During development this project consumes raitap from the parent checkout via
  `[tool.uv.sources]`. Real consumers should remove that section to fetch raitap
  from PyPI.
