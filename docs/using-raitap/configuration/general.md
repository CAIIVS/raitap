# General configuration guide

RAITAP is built on top of [Hydra](https://hydra.cc/), a powerful configuration framework for Python. It allows to configure all options via YAML files, and override them when running via the CLI.

These docs will explain just enough about Hydra to use RAITAP effectively. However, you might want to dive deeper into the [Hydra documentation](https://hydra.cc/docs/intro/).

## Guide to writing and using your own configuration

### 1. Write your configuration YAML

Hydra parses YAML files to understand which options to apply to the pipeline. Your YAML must include the `raitap_schema` defaults entry. hence, it should always start with:

```yaml
defaults:
  - raitap_schema
  - _self_

// ...your options, see below
```

Then, you can add your own options. You may find useful to refer to:

- the {doc}`global-config-options`
- the {ref}`module-specific-configurations`
- the {doc}`kitchen-sink`

If your workflow does not make it easy to use YAML files, you can rely 100% on a CLI command. See {ref}`cli-overriding` for more details.

### 2. Preview your configuration with `--help`

You can preview the final, Hydra-parsed configuration before executing it. Run the following from the same directory:

```{install-tabs}
:uv:
uv run raitap --config-name assessment --help # assuming your config is at `./assessment.yaml`

:pip:
raitap --config-name assessment --help # assuming your config is at `./assessment.yaml`
```

### 3. Execute your configuration

```{install-tabs}
:uv:
uv run raitap --config-name assessment # assuming your config is `./assessment.yaml`

:pip:
raitap --config-name assessment # assuming your config is `./assessment.yaml`
```

As mentioned in {doc}`../get-it-running`, RAITAP will then detect required dependencies and guide you in the terminal.

## Some advanced Hydra features

### Getting Hydra help

Hydra provides a `--hydra-help` flag to print the available options and their descriptions. Note that it differs from the `--help` flag, see below.

(cli-overriding)=

### CLI overriding

Hydra does not only read from YAML files. It can also parse CLI option overrides.
In the following, we override some options from the
{doc}`../../modules/transparency/configuration`.

#### 1. Discover what's available: `--help`

Pass `--help` to print every available config group + the fully composed config
for the current invocation. Useful when picking presets or sanity-checking
overrides:

```bash
uv run raitap --config-dir my-configs --config-name assessment --help
```

Output has two sections:

- **Configuration groups** — names you can pass as `<group>=<option>` (e.g.
  `reporting=pdf`, `+transparency=shap`).
- **Config** — the fully composed YAML that will be passed to the run, with
  all schema defaults expanded. Every key shown is overridable via
  `key=value` on the command line.

#### Override syntax: `key=value`, `+key=value`, `~key`

For the following, we will assume your are overriding the following YAML config:

```yaml
defaults:
  - raitap_schema
  - _self_
  - metrics: classification

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

Hydra recognises three group-override prefixes on the command line:

- `key=value` — **override** an existing key already in the config's `defaults:` list.
  Example: `metrics=detection` (works only if `metrics` is already in the defaults list).
- `+key=value` — **add** a key not yet in the defaults list.
  Example: `+reporting=html` (works even if your YAML omits `reporting`).
- `~key` — **remove** a key from the defaults list.
  Example: `~robustness.pgd` drops the named robustness assessor.

#### Overriding values

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
  - raitap_schema  # required, do not omit it, ever
  - _self_         # inserts the 2 keys below into the final config
  - transparency: shap
  - metrics: classification

experiment_name: "my-exp"
hardware: cpu

# Bundled `transparency/shap.yaml` only sets `_target_: ShapExplainer` and
# nests it under `transparency.shap`. The explainer's required fields
# (`algorithm`, `call`, `visualisers`) still need to be supplied here:
transparency:
  shap:
    algorithm: GradientExplainer
    call:
      target: 0
    visualisers:
      - _target_: ShapImageVisualiser

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

The bundled `+transparency=captum` / `+transparency=shap` stubs only set
`_target_` and nest under `transparency.captum` / `transparency.shap`; the
sweep below pairs each with the matching `algorithm` override.

```{install-tabs}
:uv:
uv run raitap --multirun +transparency=captum,shap "transparency.captum.algorithm=IntegratedGradients" "transparency.shap.algorithm=GradientExplainer" experiment_name=captum,shap

:pip:
raitap --multirun +transparency=captum,shap "transparency.captum.algorithm=IntegratedGradients" "transparency.shap.algorithm=GradientExplainer" experiment_name=captum,shap
```

Hydra expands the comma-separated values into multiple runs. To inspect where each run
writes its outputs, see {doc}`../understanding-outputs`.

### Job launcher integration (Slurm example)

See {doc}`../job-launcher` for more details.
