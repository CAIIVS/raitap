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

#### Inherit raitap's schema defaults: `defaults: [raitap_schema, _self_]`

Every consumer config should start with the `raitap_schema` defaults entry. It
binds raitap's `AppConfig` dataclass schema to your config, so:

- **Unset optional fields** inherit safe defaults (e.g. `reporting.sample_selection`
  is `null`, `reporting.multirun_report` is `true`) — no crashes when pipeline
  code reads them.
- **Required fields** stay `MISSING` and fail loudly at composition time if you
  forget them (e.g. `metrics._target_`).
- **Types are enforced** — pass `reporting.multirun_report=42` and Hydra rejects
  it before the run starts.

Minimal example:

```yaml
defaults:
  - raitap_schema     # bind AppConfig schema
  - _self_            # apply this file's overrides on top
  - reporting: html   # compose bundled reporting/html.yaml
  - metrics: classification

experiment_name: my-exp
hardware: cpu

model:
  source: vit_b_32

data:
  name: imagenet_samples
  source: imagenet_samples

transparency:
  default:
    _target_: CaptumExplainer
    algorithm: IntegratedGradients
    call:
      target: 0
    visualisers:
      - _target_: CaptumImageVisualiser

robustness:
  pgd:
    _target_: TorchattacksAssessor
    algorithm: PGD
    constructor:
      eps: 0.03
      alpha: 0.005
      steps: 10
    visualisers:
      - _target_: ImagePairVisualiser
```

Skip `raitap_schema` only if you're hand-rolling every field — it's never wrong
to include it.

#### Override syntax: `key=value`, `+key=value`, `~key`

Hydra recognises three group-override prefixes on the command line:

- `key=value` — **override** an existing key already in the config's `defaults:` list.
  Example: `reporting=pdf` (works only if `reporting` is already in the defaults list).
- `+key=value` — **add** a key not yet in the defaults list.
  Example: `+reporting=html` (works even if your YAML omits `reporting`).
- `~key` — **remove** a key from the defaults list.
  Example: `~robustness.pgd` drops the named robustness assessor.

Bundled raitap groups (`reporting/html`, `transparency/captum`,
`robustness/torchattacks`, `metrics/classification`, `tracking/mlflow`) are
auto-discovered from the installed package via raitap's `SearchPathPlugin`, so
they work from any user config directory without manual `hydra: searchpath:`
declarations.

Example — run an external config and bolt on bundled HTML reporting:

```bash
uv run raitap --config-dir my-configs --config-name assessment +reporting=html
```


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
  - transparency: shap            # bundled group stub from raitap (see SearchPathPlugin note below)
  - metrics: classification       # bundled group stub from raitap

experiment_name: "my-exp"
hardware: cpu

# Inline model + data — RAITAP does not ship `data=` or `model=` presets, so
# define them in your own config (or reference your own group files).
model:
  source: resnet50  # built-in torch model, see the Model module docs

data:
  source: ./my-dataset
```

Hydra composition is cascading top-down. Hence, you might want to control the
order of composition. This can be achieved using the `_self_` keyword (note the single underscores).

```yaml
defaults:
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
uv run raitap --multirun +transparency=captum,shap experiment_name=captum,shap

:pip:
raitap --multirun +transparency=captum,shap experiment_name=captum,shap
```

Hydra expands the comma-separated values into multiple runs. To inspect where each run
writes its outputs, see {doc}`../understanding-outputs`.

### Job launcher integration (Slurm example)

See {doc}`../job-launcher` for more details.
