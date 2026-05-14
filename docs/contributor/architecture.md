# Architecture

The page explains the high level architecture of the `raitap` package.

```{mermaid}
flowchart TB
  A[uv run raitap …] --> B[raitap.cli.main]
  B -->|tracking stop?| Z[run_stop_command]
  B -->|--demo?| C1[load bundled demo.yaml]
  B -->|else| C2[parse user args]
  C1 --> D[raitap.deps.bootstrap.maybe_bootstrap]
  C2 --> D
  D -->|missing extras?| D1[uv sync + re-exec]
  D1 -->|re-exec| D
  D --> E[raitap.pipeline.__main__]
  E --> F["@hydra.main composes config"]
  F --> G[raitap.pipeline.pipeline.run]
  G --> H[forward · metrics · transparency · robustness]
  H --> I[reporting + tracking]
```

## Module map

| Module                      | Responsibility                                                                                                                                                                                                                          |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `raitap.cli`                | Console-script entry. Handles `tracking stop` subcommand, `--demo` flag, bare-invocation help frame, then delegates to `raitap.deps.bootstrap`.                                                                                         |
| `raitap.deps`               | Pre-pipeline dependency inference + auto-sync. Walks the composed Hydra config, picks backend + adapter extras for the host, then re-execs via `uv run` (dev) or prints an `uv add` / `pip install` plan (installed). Stays torch-free. |
| `raitap.pipeline.__main__`  | `@hydra.main` entry. Composes the config (incl. bundled `raitap_schema` structured-config), then dispatches to `pipeline.run`.                                                                                                          |
| `raitap.pipeline.pipeline`  | Orchestrates the assessment run: model + data load, forward pass, metrics, transparency explainers, robustness assessors, optional tracking.                                                                                            |
| `raitap.configs.searchpath` | `RaitapSearchPathPlugin` — appends `pkg://raitap.configs` to Hydra's search path so external configs resolve bundled group presets without manual `hydra.searchpath` wiring. Surfaced via the `hydra_plugins` namespace package.        |
| `raitap.reporting`          | Report builder + HTML / PDF reporters + multirun sweep callback.                                                                                                                                                                        |
| `raitap.tracking`           | Tracker base class + MLflow adapter + `tracking stop` subcommand.                                                                                                                                                                       |
