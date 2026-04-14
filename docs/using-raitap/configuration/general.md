# General configuration guide

RAITAP is built on top of [Hydra](https://hydra.cc/), a powerful configuration framework for Python. It allows to configure all options via YAML files, and override them when running via the CLI.

These docs will explain just enough about Hydra to use RAITAP effectively. However, you might want to dive deeper into the [Hydra documentation](https://hydra.cc/docs/intro/).

## Guide to writing and using your own configuration

### 1. Write your configuration YAML

Hydra parses YAML files to understand which options to apply to the pipeline. Create a YAML file with the options you need.
You may find useful to refer to:

- the {doc}`global-config-options`
- the {ref}`module-specific-configurations`
- the {doc}`kitchen-sink`

If your workflow does not make it easy to use YAML files, you can rely 100% on a CLI command. See {ref}`cli-overriding` for more details.

### 2. Preview your configuration

You can preview the final, Hydra-parsed configuration before executing it. Run the following from the same directory:

```{install-tabs}
:uv:
uv run raitap --config-name assessment --cfg job # assuming your config is at `./assessment.yaml`

:pip:
raitap --config-name assessment --cfg job # assuming your config is at `./assessment.yaml`
```

### 3. Execute your configuration

```{install-tabs}
:uv:
uv run raitap --config-name assessment # assuming your config is `./assessment.yaml`

:pip:
raitap --config-name assessment # assuming your config is `./assessment.yaml`
```

## Some advanced Hydra features

(cli-overriding)=

### CLI overriding

Hydra does not only read from YAML files. It can also parse CLI option overrides.
In the following, we override some options from the
{doc}`../../modules/transparency/configuration`.

You can either set individual options:

```{install-tabs}
:uv:
uv run raitap --config-name assessment hardware=cpu transparency.myexplainer1.call.target=0

:pip:
raitap --config-name assessment hardware=cpu transparency.myexplainer1.call.target=0
```

Or override an entire nested value at once:

```{install-tabs}
:uv:
uv run raitap --config-name assessment "transparency.captum_saliency.visualisers=[{_target_: CaptumImageVisualiser, call: {show_sample_names: true}}]"

:pip:
raitap --config-name assessment "transparency.captum_saliency.visualisers=[{_target_: CaptumImageVisualiser, call: {show_sample_names: true}}]"
```

(composing-yaml-files)=

### Composing YAML files

Hydra allows you to compose multiple YAML files into a single configuration.
This is useful to avoid repeating the same options in multiple files.

The main mechanism for this is the `defaults` list.

```yaml
# assessment.yaml
defaults:
  - _self_ # inserts experiment_name and hardware from the current file into the final config
  - model: resnet50 # imports the other YAML file, see below
  - data: isic2018
  - transparency: shap_gradient
  - metrics: classification

experiment_name: "my-exp"
hardware: cpu

# resnet50.yaml
source: resnet50 # built-in torch model, see the Model module docs
```

Hydra composition is cascading top-down. Hence, you might want to control the
order of composition. This can be achieved using the `_self_` keyword (note the single underscores).

```yaml
defaults:
  - model: resnet50
  - model: vitb32
  - _self_

model:
  source: "./my-custom-model.onnx"
```

In the above example, the final config will use the custom ONNX model, because `_self_` is applied last.

### Batch runs

Hydra can execute multiple runs from a single command using `--multirun`.
This is useful when you want to compare several presets or override values in one go.

```{install-tabs}
:uv:
uv run raitap --multirun transparency=demo,shap_gradient experiment_name=demo,shap

:pip:
raitap --multirun transparency=demo,shap_gradient experiment_name=demo,shap
```

Hydra expands the comma-separated values into multiple runs. To inspect where each run
writes its outputs, see {doc}`../understanding-outputs`.

### Job launcher integration (Slurm example)

See {doc}`../job-launcher` for more details.
