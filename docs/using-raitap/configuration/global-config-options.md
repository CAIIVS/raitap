# Global configuration

This section describes options that impact all modules.

## Options

```{config-options}
:option: hardware
:allowed: "cpu", "gpu"
:default: "gpu"
:description: Forces execution on the specified hardware. If the GPU is unavailable
  on the machine, RAITAP falls back to CPU and emits a CLI warning.

:option: experiment_name
:allowed: string
:default: "Experiment"
:description: Name of the experiment (assessment run).

:option: hydra.run.dir
:allowed: string
:default: "./outputs/<date>/<time>"
:description: Directory where Hydra stores the run outputs. If not specified,
  Hydra creates a timestamped directory under `./outputs` relative to the
  terminal working directory where RAITAP was launched. If you want to forward outputs to your tracking software, see [Tracking](../../modules/tracking/configuration.md).
```

## YAML example

```yaml
hydra:
  run:
    dir: "./custom-outputs-dir"

hardware: "gpu"
experiment_name: "My Experiment"
```

## CLI override example

```{install-tabs}
:uv:
uv run raitap hardware=cpu experiment_name="My_Experiment"

:pip:
raitap hardware=cpu experiment_name="My_Experiment"
```

For module-specific options and examples, refer to the standalone module pages listed above.
