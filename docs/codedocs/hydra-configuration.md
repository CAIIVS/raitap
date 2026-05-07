---
title: "Hydra Configuration"
description: "Learn how RAITAP uses Hydra groups, structured config dataclasses, and CLI overrides."
---

Hydra is the configuration backbone of RAITAP. The structured schema in `src/raitap/configs/schema.py` defines the stable top-level shape, while YAML presets in `src/raitap/configs/` choose concrete model, data, transparency, metrics, tracking, and reporting settings for a run.

## What This Concept Solves

Without a config layer, responsible-AI assessments turn into a long Python script full of environment-specific paths and framework-specific constructor calls. RAITAP moves those decisions into Hydra groups so you can compose a run from reusable presets and then override only the pieces that change.

This concept relates directly to the run pipeline in `/docs/architecture`, because `raitap.run.__main__` and `raitap.run.run()` both assume a fully resolved `AppConfig`. It also relates to explainers and metrics, because the factories in `src/raitap/transparency/factory.py` and `src/raitap/metrics/factory.py` both interpret `_target_` values from that config.

## How It Works Internally

`register_configs()` in `src/raitap/configs/utils.py` registers `AppConfig` with Hydra's `ConfigStore`. The top-level CLI in `src/raitap/run/__init__.py` calls `register_configs()` on import so the config schema is available before any run starts.

The default job config in `src/raitap/configs/config.yaml` composes:

- `transparency: demo`
- `model: vit_b32`
- `data: isic2018`
- `metrics: classification`
- `reporting: pdf`

The schema itself is intentionally small:

```python
@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    transparency: dict[str, Any] = field(default_factory=lambda: {"default": TransparencyConfig()})
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    reporting: ReportingConfig | None = None
    hardware: str = "gpu"
    experiment_name: str = "Experiment"
```

That split matters. The dataclass gives RAITAP a predictable shape, while the YAML groups provide real runtime content. The helper `resolve_target()` in `src/raitap/configs/utils.py` then expands short class names such as `CaptumExplainer` into `raitap.transparency.CaptumExplainer` and `ClassificationMetrics` into `raitap.metrics.ClassificationMetrics`.

## Basic Usage

This example uses the built-in config directory exactly the way the CLI does.

```python
from hydra import compose, initialize_config_dir
from pathlib import Path

config_dir = Path("src/raitap/configs").resolve()

with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
    cfg = compose(
        config_name="config",
        overrides=[
            "hardware=cpu",
            "model=resnet50",
            "data=imagenet_samples",
            "reporting=disabled",
        ],
    )

print(cfg.model.source)
print(cfg.data.source)
```

## Advanced Usage

A more realistic pattern is composing a local assessment file and overriding nested transparency values at the CLI boundary:

```yaml
# assessment.yaml
defaults:
  - _self_
  - model: resnet50
  - data: imagenet_samples
  - transparency: shap_gradient
  - metrics: classification
  - reporting: disabled

experiment_name: "audit-cpu"
hardware: cpu
```

```bash
uv run raitap --config-name assessment \
  transparency.shap_gradient.call.target=0 \
  transparency.shap_gradient.raitap.batch_size=1
```

The nested override lands in the same dictionary consumed later by `_parse_explainer_config()` in `src/raitap/transparency/factory.py`.

<Callout type="warn">Do not put RAITAP-owned runtime options such as `batch_size`, `sample_names`, or `show_progress` under `call:` unless you are relying on the factory's migration warning path. `src/raitap/transparency/factory.py` explicitly treats them as `raitap:` options and only keeps the migration logic to soften bad configs, not to define the main API.</Callout>

<Accordions>
<Accordion title="Why RAITAP uses short _target_ names">
Short `_target_` values keep user config readable and reduce duplication across presets. The trade-off is that RAITAP must maintain predictable namespace prefixes in `resolve_target()`, so a bare `CaptumExplainer` only works because the factory knows to prepend `raitap.transparency.`. That is a good fit here because the project owns the extension points and wants concise configs. If you need a third-party class, you can still use a fully qualified import path and bypass prefix expansion.

```yaml
metrics:
  _target_: raitap.metrics.ClassificationMetrics
  task: multiclass
  num_classes: 1000
```
</Accordion>
<Accordion title="Why the schema stays shallow">
The schema dataclasses intentionally do not try to model every framework-specific argument. That keeps `AppConfig` stable even when Captum, SHAP, or MLflow-specific knobs change, but it also means some nested blocks are typed as `dict[str, Any]`. RAITAP compensates for that with module-specific validators such as `_validate_explainer_top_level_keys()` and `_validate_raitap_keys()` in `src/raitap/transparency/factory.py`. The result is more flexible than a deeply nested rigid schema, but validation is distributed across modules instead of centralized in one dataclass tree.

```yaml
transparency:
  shap_deep:
    _target_: ShapExplainer
    algorithm: DeepExplainer
    constructor: {}
    call:
      target: 0
```
</Accordion>
</Accordions>

Hydra is the reason RAITAP can treat the CLI, tests, and programmatic runs as the same system. If you understand this page, the rest of the package becomes much easier to predict.
