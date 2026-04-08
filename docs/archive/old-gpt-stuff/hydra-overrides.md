# Use Hydra Overrides

RAITAP's main interface is Hydra composition plus CLI overrides. The top-level config is
defined in `src/raitap/configs/config.yaml`, and individual groups are selected by name.

## Switch config groups

Pick a different preset by setting the config-group key on the CLI:

```bash
uv run raitap model=resnet50 data=imagenet_samples transparency=shap_gradient
```

The built-in defaults are:

- `transparency=demo`
- `model=vit_b32`
- `data=isic2018`
- `metrics=classification`
- `tracking=null`

## Override a single field

Override one concrete config value directly:

```bash
uv run raitap hardware=cpu
uv run raitap experiment_name=demo_cpu
uv run raitap model.source=models/resnet50.pth
```

## Override nested transparency call arguments

Transparency configs separate constructor-time arguments from call-time arguments:

- `constructor`: values used when the explainer or visualiser object is instantiated
- `call`: values passed each time the explainer or visualiser is executed

For example, change the target class used by the default demo preset:

```bash
uv run raitap transparency.captum_ig.call.target=2
```

Select a different algorithm inside a named transparency preset:

```bash
uv run raitap transparency.captum_ig.algorithm=GradientShap
```

## Override a list of visualisers

Hydra accepts full list replacement syntax:

```bash
uv run raitap "transparency.captum_ig.visualisers=[{_target_: CaptumImageVisualiser}]"
```

Because the list is replaced as a whole, use this pattern when you want full control
over which visualisers are attached to a preset.

## Inspect the final config before running

Use Hydra's config inspection mode whenever you are not sure what the final composed job
looks like:

```bash
uv run raitap --cfg job
```

This is the safest way to verify combined group selections and nested overrides before
launching a real assessment.
