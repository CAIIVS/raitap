---
title: "Global configuration"
description: "This section describes options that impact all modules."
myst:
  html_meta:
    "description": "This section describes options that impact all modules."
---

# Global configuration

This section describes options that impact all modules.

## Options

```{config-options}
:option: hardware
:allowed: "cpu", "gpu"
:default: "gpu"
:description: Forces execution on the specified hardware. The config-level
  enum is only `cpu` / `gpu`; when set to `gpu`, RAITAP probes for CUDA
  (NVIDIA), then XPU (Intel), then MPS (Apple, planned) and binds to the
  first one it finds. If none is available, RAITAP falls back to CPU and
  emits a CLI warning. The probed backend must match the installed extra —
  `gpu` on a CUDA host needs `torch-cuda` / `onnx-cuda`, `gpu` on an Intel
  host needs `torch-intel` / `onnx-intel`. See
  {ref}`execution-dependencies` for the extras matrix.

:option: experiment_name
:allowed: string
:default: "Experiment"
:description: Name of the experiment (assessment run).

:option: hydra.run.dir
:allowed: string
:default: "./outputs/<date>/<time>"
:description: Directory where Hydra stores the run outputs. If not specified,
  Hydra creates a timestamped directory under `./outputs` relative to the
  terminal working directory where RAITAP was launched. If you want to forward outputs to your tracking software, see {doc}`../../modules/tracking/configuration`.
```

## YAML example

```{config-tabs}
:yaml:
hydra:
  run:
    dir: "./custom-outputs-dir"

hardware: "gpu"
experiment_name: "My Experiment"

:python:
from raitap import AppConfig, Hardware

# Python users construct AppConfig directly and call ``run(config)``;
# ``hydra.run.dir`` is a CLI/YAML-only knob (set the output directory in
# your own code instead).
config = AppConfig(
    hardware=Hardware.gpu,
    experiment_name="My Experiment",
)
```

## CLI override example

```{install-tabs}
:uv:
uv run raitap hardware=cpu experiment_name="My_Experiment"

:pip:
raitap hardware=cpu experiment_name="My_Experiment"
```

For module-specific options and examples, refer to {ref}`module-specific-configurations`.
