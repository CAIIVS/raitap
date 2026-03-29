# Configuration guide

RAITAP uses [Hydra](https://hydra.cc/) for configuration management. All settings can be overridden from the CLI — no YAML files required.

## Quick start

```bash
# Run with defaults (ViT-B/32, ISIC2018, Captum/IntegratedGradients)
uv run raitap

# Switch config groups
uv run raitap model=resnet50 data=imagenet_samples transparency=shap

# Override a specific field
uv run raitap transparency.algorithm=Saliency
```

## Config groups

Select built-in presets by name on the CLI.

### `model`

| Name       | Description                          |
| ---------- | ------------------------------------ |
| `vit_b32`  | ViT-B/32, ImageNet weights (default) |
| `resnet50` | ResNet-50, ImageNet weights          |
| `custom`   | Template for a custom model path     |

### `data`

| Name                 | Description                             |
| -------------------- | --------------------------------------- |
| `isic2018`           | ISIC 2018 skin lesion dataset (default) |
| `imagenet_samples`   | Sample ImageNet images (4 classes)      |
| `malaria`            | Malaria cell image dataset              |
| `udacityselfdriving` | Udacity self-driving car dataset        |

### `transparency`

| Name     | Explainer         | Default algorithm     | Default visualiser      |
| -------- | ----------------- | --------------------- | ----------------------- |
| `captum` | `CaptumExplainer` | `IntegratedGradients` | `CaptumImageVisualiser` |
| `shap`   | `ShapExplainer`   | `GradientExplainer`   | `ShapImageVisualiser`   |

### Optional: sample names in visualisations

You can optionally show sample names in visual outputs via visualiser `call` arguments.

- `show_sample_names` defaults to `false`
- names come from loaded sample IDs (filename stem, extension removed)
- if name count and batch size differ, names are trimmed to the plotted batch

Example override on the CLI:

```bash
uv run raitap transparency=gradcam "transparency.captum_saliency.visualisers=[{_target_: CaptumImageVisualiser, call: {show_sample_names: true}}]"
```

Equivalent YAML style:

```yaml
transparency:
  captum_saliency:
    _target_: CaptumExplainer
    algorithm: Saliency
    visualisers:
      - _target_: CaptumImageVisualiser
        call:
          show_sample_names: true
```

Notes:

- Visualisers with native support (for example `CaptumImageVisualiser`) render per-sample titles.
- Other visualisers fall back to a figure title using the first name (format: `first (+N)`).

## Custom model and data

Override `model.source` and `data.source` directly:

```bash
uv run raitap model.source=models/resnet.pth data.source=data/my_dataset
```

- `model.source` — local `.pth` file path or a built-in name (`resnet50`, `vit_b_32`)
- `data.source` — local directory path or a named sample set

Optional label fields on `data` enable metric runs against ground truth:

- `data.labels_source` — path/URL to CSV, TSV, or Parquet labels
- `data.labels_id_column` — sample-id column for filename matching (e.g. `image`)
- `data.labels_column` — direct class-index column (optional)
- `data.labels_encoding` — parsing strategy: `index`, `one_hot`, `argmax`

### Supported label data formats

`data.labels_source` currently supports tabular files only:

- `.csv`
- `.tsv`
- `.parquet`

RAITAP accepts the following label layouts:

1. Single class-index column
   - Set `data.labels_column=<column_name>`
   - Values must be numeric class indices (for example `0, 1, 2, ...`)
2. One-hot or score matrix across multiple numeric columns
   - Do not set `data.labels_column`
   - RAITAP uses `argmax` across numeric label columns

`data.labels_encoding` behavior:

- `index`: expects a single numeric label column (or explicit `labels_column`)
- `one_hot`: expects multiple numeric columns and resolves labels via `argmax`
- `argmax`: resolves labels via `argmax` when multiple numeric columns are present

Sample-to-label matching:

- If `data.labels_id_column` is set (or auto-detected as one of `image`, `filename`, `file`, `id`, `name`), labels are matched to sample filenames by stem (extension ignored)
- If no ID column is available, row-order matching is used and label count must equal sample count

Fallback behavior (metrics still run, but against predictions):

- Missing labels file rows for some samples
- Duplicate IDs in the labels file
- Label count mismatch between loaded samples and labels
- Empty labels file


Any config field can be overridden:

```bash
# Change algorithm
uv run raitap transparency.algorithm=Saliency

# Set experiment name
uv run raitap experiment_name=audit_2026_Q1

# Combine overrides
uv run raitap model=resnet50 data=imagenet_samples transparency.algorithm=GradientShap experiment_name=demo
```

## Outputs

Each run writes to `outputs/<date>/<time>/`:

```text
outputs/
└── 2026-02-28/
    └── 14-30-45/
        ├── attributions.pt        # Attribution tensor (torch.Tensor)
        ├── <VisualiserName>.png   # One file per visualiser
        └── metadata.json          # Config snapshot
```

`metadata.json` records the explainer class, algorithm, visualisers, and experiment name.

## Batch runs (multirun)

Run multiple configurations in one command:

```bash
uv run raitap --multirun transparency=captum,shap experiment_name=cmp_captum,cmp_shap
```

## Dry run

Inspect the resolved config without running:

```bash
uv run raitap --cfg job
```

## Python API

For programmatic use, `explain()` is the single entry point:

```python
from raitap.transparency import explain

result = explain(config, model, inputs)
attributions = result["attributions"]    # torch.Tensor
figures      = result["visualisations"]  # dict[str, matplotlib.figure.Figure]
run_dir      = result["run_dir"]         # pathlib.Path
```

`config` must expose `.transparency`, `.experiment_name`, and `.fallback_output_dir` attributes (i.e. an `AppConfig` instance or a compatible dataclass/namespace).
