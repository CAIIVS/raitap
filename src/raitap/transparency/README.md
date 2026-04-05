# RAITAP Transparency Module

A flexible explainability module for PyTorch models wrapping SHAP and Captum.

## Quick start

```bash
# Run via CLI (no Python required)
uv run raitap model=resnet50 transparency=captum
uv run raitap transparency=shap transparency.algorithm=GradientExplainer
```

```python
from raitap.transparency import explain

result = explain(config, model, inputs, target=0)
# result["attributions"]   → torch.Tensor
# result["visualisations"] → dict[str, matplotlib.figure.Figure]
# result["run_dir"]        → pathlib.Path
```

## Architecture

```
CLI / Config
  └── explain()              # factory.py — single entry point
        ├── Hydra instantiate(_target_)
        │     ├── CaptumExplainer / ShapExplainer
        │     └── CaptumImageVisualiser / ShapImageVisualiser / …
        └── outputs/<date>/<time>/
              ├── attributions.pt
              ├── <VisualiserName>.png
              └── metadata.json
```

## Components

### Explainers (`explainers/`)

- `CaptumExplainer` — wraps any `captum.attr.*` algorithm via `getattr`
- `ShapExplainer` — wraps any `shap.*Explainer` algorithm

Both implement `compute_attributions(model, inputs, **kwargs) → torch.Tensor`.

### Visualisers (`visualisers/`)

| Class                        | Framework | Output                                                    |
| ---------------------------- | --------- | --------------------------------------------------------- |
| `CaptumImageVisualiser`      | Captum    | image heatmap overlay                                     |
| `CaptumTimeSeriesVisualiser` | Captum    | time-series attribution plot                              |
| `CaptumTextVisualiser`       | Captum    | text token importance                                     |
| `ShapImageVisualiser`        | SHAP      | pixel-level SHAP (GradientExplainer / DeepExplainer only) |
| `ShapBarVisualiser`          | SHAP      | mean absolute bar chart                                   |
| `ShapBeeswarmVisualiser`     | SHAP      | beeswarm summary                                          |
| `ShapWaterfallVisualiser`    | SHAP      | per-sample waterfall                                      |
| `ShapForceVisualiser`        | SHAP      | per-sample force plot                                     |
| `TabularBarChartVisualiser`  | any       | plain bar chart for tabular data                          |

Visualiser compatibility with specific algorithms is declared via the `compatible_algorithms` class attribute (`frozenset`). An empty frozenset means compatible with all algorithms.

### Config (`configs/transparency/`)

Selection is done by `_target_` key — Hydra instantiates the class directly:

```yaml
# configs/transparency/captum.yaml
_target_: CaptumExplainer
algorithm: IntegratedGradients
visualisers:
  - _target_: CaptumImageVisualiser
```

Explainer `call` options can also control batching behaviour. If you set `batch_size` or `max_batch_size`, progress bars are enabled by default and can be disabled with `show_progress: false`.

```yaml
call:
  batch_size: 8
  show_progress: false
  progress_desc: My explainer batches
```

## Supported algorithms

Any algorithm accessible via `captum.attr.<name>` or `shap.<name>Explainer` works without code changes:

```bash
uv run raitap transparency.algorithm=Saliency
uv run raitap transparency=shap transparency.algorithm=KernelExplainer
```

## Testing

```bash
uv run pytest tests/transparency/ -v
```

Test files:

- `test_captum_explainer.py` — Captum wrapper
- `test_shap_explainer.py` — SHAP wrapper
- `test_methods.py` — `compatible_algorithms` contracts
- `test_visualisers.py` — visualiser implementations
- `test_integration.py` — end-to-end via `explain()`

## Dependencies

- `torch>=2.0.0`
- `matplotlib>=3.5.0`
- `captum>=0.7.0` (optional, for Captum explainers/visualisers)
- `shap>=0.46.0` (optional, for SHAP explainers/visualisers)
