# NoiseTunnel (SmoothGrad) smoke config

Minimal end-to-end check for captum NoiseTunnel support (#269). Wraps `Saliency`
in `NoiseTunnel` to produce SmoothGrad, next to bare `Saliency` for contrast.

Self-contained: uses the bundled `imagenet_samples` data and the torchvision
`vit_b_32` model, so no local artifacts are needed (first run downloads the ViT
weights). NoiseTunnel is stochastic, so the run prints the reproducibility
caveat and the report flags it.

## Run

```bash
uv run --extra torch-intel --extra captum --extra metrics --extra reporting \
  raitap --config-dir contributor-configs/noisetunnel-smoothgrad --config-name assessment
```

The HTML report lands under `outputs/<date>/<time>/reports/`.

## Knobs

- `transparency.smoothgrad_over_saliency.constructor.base_algorithm` — wrapped
  method. Must be a non-layer gradient method: `Saliency` or `IntegratedGradients`.
- `call.nt_type` — `smoothgrad` | `smoothgrad_sq` | `vargrad`.
- `call.nt_samples` — noisy copies averaged (higher = smoother, slower).
- `call.stdevs` — Gaussian noise scale.
