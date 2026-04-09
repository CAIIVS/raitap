# Creating & running your own configuration

This page explains how to configure RAITAP for your use case, and includes a [kitchen-sink example](configuration.md#kitchen-sink-example).

## General configuration principles

### Hydra

RAITAP is built on top of [Hydra](https://hydra.cc/), a powerful configuration framework for Python. It allows to configure all options via YAML files, and override them when running via the CLI.

These docs will explain just enough about Hydra to use RAITAP effectively. However, you might want to dive deeper into the [Hydra documentation](https://hydra.cc/docs/intro/).

### RAITAP modules

A RAITAP config consists of several modules, each of which is responsible for a specific part of the pipeline.

Module-specific configuration now lives in standalone module pages:

- [Model](../modules/model/configuration.md)
- [Data](../modules/data/configuration.md)
- [Transparency](../modules/transparency/configuration.md)
- [Metrics](../modules/metrics/configuration.md)
- [Tracking](../modules/tracking/configuration.md)

## High-level configuration

This section describes high-level options that impact all modules.

### All options

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
```

### YAML example

```yaml
hardware: "gpu"
experiment_name: "My Experiment"
```

### CLI override example

```{install-tabs}
:uv:
uv run raitap hardware=cpu experiment_name="My_Experiment"

:pip:
raitap hardware=cpu experiment_name="My_Experiment"
```

For module-specific options and examples, refer to the standalone module pages listed above.

## Kitchen-sink example

The example below shows a complete configuration with all top-level modules populated.

```yaml
hardware: "gpu"
experiment_name: "My Experiment"

model:
  source: "resnet50"

data:
  name: "my-dataset"
  description: "Internal validation set"
  source: "./data/images"
  labels:
    source: "./data/labels.csv"
    id_column: "image"
    column: "label"
    encoding: "index"

transparency:
  default:
    _target_: "CaptumExplainer"
    algorithm: "IntegratedGradients"
    constructor: {}
    call: {}
    visualisers:
      - _target_: "CaptumImageVisualiser"

metrics:
  _target_: "ClassificationMetrics"
  task: "multiclass"
  num_classes: 7

tracking:
  _target_: "MLFlowTracker"
  output_forwarding_url: null
  log_model: false
  open_when_done: false
```
