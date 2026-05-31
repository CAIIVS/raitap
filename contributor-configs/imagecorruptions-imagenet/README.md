# ImageNet-C average-case robustness

Average-case robustness demo: the `imagecorruptions` (ImageNet-C) adapter
applies one common corruption at one severity to the bundled `imagenet_samples`
subset, measures accuracy under the corruption, and renders a clean-vs-corrupted
accuracy chart in an HTML report.

Self-contained — no external model or data files. Uses a torchvision built-in
ResNet-18 and the bundled `imagenet_samples` dataset.

## Run

From the repo root:

```bash
contributor-configs/imagecorruptions-imagenet/run.sh           # POSIX
contributor-configs\imagecorruptions-imagenet\run.ps1          # Windows
```

Or directly:

```bash
uv run --extra torch-intel --extra imagecorruptions --extra metrics --extra reporting \
  raitap --config-dir contributor-configs/imagecorruptions-imagenet --config-name assessment
```

The HTML report lands under the Hydra run dir (`outputs/<date>/<time>/reports/`).

## Knobs

- `robustness.gaussian_noise.algorithm` — any of the 15 ImageNet-C corruptions
  (`gaussian_noise`, `shot_noise`, `impulse_noise`, `defocus_blur`, `glass_blur`,
  `motion_blur`, `zoom_blur`, `snow`, `frost`, `fog`, `brightness`, `contrast`,
  `elastic_transform`, `pixelate`, `jpeg_compression`).
- `robustness.gaussian_noise.severity` — `1..5`.
- `raitap.ci_method` / `raitap.ci_level` — binomial CI for the corrupted-accuracy
  estimate.

## What's in the report

- Clean vs corrupted accuracy bars with a confidence-interval whisker
  (`CorruptionAccuracyVisualiser`), annotated with corruption name, severity, N.
- Metrics on the presentation subset — treat as a sanity check, not a benchmark.
