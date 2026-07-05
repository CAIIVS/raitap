# Quantus explanation-quality grading

Grades attribution quality with [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus)
as a post-step of the transparency phase. Runs Captum IntegratedGradients on the
bundled `imagenet_samples` subset with a torchvision `vit_b_32` model, then a
config-driven `QuantusEvaluator` scores the resulting attributions.

Self-contained — no external model or data files.

## Run

From the repo root:

```bash
contributor-configs/quantus-eval/run.sh           # POSIX
contributor-configs\quantus-eval\run.ps1          # Windows
```

Or directly:

```bash
uv run --extra torch-cpu --extra captum --extra quantus --extra metrics --extra html \
  raitap --config-dir contributor-configs/quantus-eval --config-name assessment
```

Note: a bare `uv run raitap` from inside the raitap repo fails with a
self-dependency error (the deps bootstrap tries `uv add raitap[...]`). Passing
the extras explicitly, as above, uses `uv sync` instead and runs.

## Knobs

- `transparency.ig.evaluation.metrics` — which Quantus metrics to compute.
  Attribution-only metrics (`sparseness`, `complexity`) always run; model-based
  metrics (`faithfulness_correlation`) need the model; re-explain metrics
  (robustness, randomisation) need an attribution explainer; localisation
  metrics need segmentation masks (none here, so they are skipped with a
  recorded reason).
- `transparency.ig.evaluation.constructor.<metric>` — per-metric Quantus
  constructor kwargs, e.g. `faithfulness_correlation: {nr_runs: 10, subset_size: 32}`.
- `transparency.ig.evaluation.raitap.softmax` — pass model outputs through
  softmax before grading (`false` by default; models emit logits).

## What you get

Per graded explanation: a score per metric (with aggregate) plus a list of
skipped metrics and the reason each was skipped. Scores are carried on
`TransparencyPhaseResult.evaluations` and logged to the configured tracker; a
`ScoreBarVisualiser` renders a per-metric bar chart.
