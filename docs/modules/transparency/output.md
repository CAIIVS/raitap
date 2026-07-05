---
title: "Output"
description: "If an explainer uses more than one visualiser, RAITAP writes one PNG per visualiser using the pattern <VisualiserClassName><index>.png."
myst:
  html_meta:
    "description": "If an explainer uses more than one visualiser, RAITAP writes one PNG per visualiser using the pattern <VisualiserClassName><index>.png."
---

# Output

```text
└── transparency/
    ├── captum_ig/                           # Output for the `captum_ig` explainer
    │   ├── attributions.pt                  # Raw attribution tensor
    │   ├── payloads/                        # Optional; only when the explainer emits structured payloads
    │   │   └── convergence_delta.pt         # One file per structured payload
    │   ├── CaptumImageVisualiser_0.png      # Visualisation written by the first visualiser
    │   └── metadata.json                    # Explainer metadata plus separate call_kwargs
    └── captum_saliency/                     # One subdirectory per named explainer
        ├── attributions.pt                  # Raw attribution tensor
        ├── CaptumImageVisualiser_0.png      # Visualisation written by the first visualiser
        └── metadata.json                    # Explainer metadata plus separate call_kwargs
```

If an explainer uses more than one visualiser, RAITAP writes one PNG per
visualiser using the pattern `<VisualiserClassName>_<index>.png`.

`metadata.json` stores typed explanation semantics, including scope, payload
kind, method families, sample selection, input metadata, output-space metadata,
and `semantics.seeding` (`deterministic`, `global_rng`, or `self_seeded`; e.g.
SHAP `GradientExplainer` and `KernelExplainer` are `global_rng`,
`PermutationExplainer` is `self_seeded`). `global_rng` and `self_seeded` drive
the run-level reproducibility caveat: a pinned `seed` config covers
`global_rng` methods, but `self_seeded` methods still name themselves as not
reproducible (see {doc}`/using-raitap/understanding-outputs`). It also keeps two
separate runtime buckets:

- `kwargs`: RAITAP-owned metadata used for downstream visualisation, such as
  `sample_names` and `show_sample_names`
- `call_kwargs`: library invocation parameters such as `target`, `baselines`,
  `background_data`, or `nsamples`

`call_kwargs` is a best-effort JSON summary of the library invocation. Scalar
values are stored directly, while tensor-like values are summarized rather than
embedded verbatim so `metadata.json` stays lightweight and readable.

## Structured payloads

Some explainers emit additive diagnostics alongside the principal attribution,
such as the Captum `convergence_delta` (from `return_convergence_delta`) or the
SHAP `base_value`. Each one is persisted to `payloads/<name>.pt` and described in
`metadata.json` under a `structured_payloads` array, one entry per payload:

```json
"structured_payloads": [
  {
    "name": "convergence_delta",
    "kind": "convergence_delta",
    "storage": "tensor",
    "file": "payloads/convergence_delta.pt",
    "shape": [16],
    "dtype": "torch.float32"
  }
]
```

The `payloads/` directory and the `structured_payloads` array are written only
when an explainer produces such payloads; explainers that emit a plain
attribution tensor omit both.

## Output spaces

Most attributions land in `INPUT_FEATURES` (pixels, tabular columns) or `IMAGE_SPATIAL_MAP` (CAM methods). Non-CAM `Layer*` captum methods (LayerConductance, LayerIntegratedGradients, LayerActivation, LayerDeepLift, LayerGradientXActivation, LayerLRP) produce a raw attribution tensor aligned to the chosen hidden layer rather than the input, recorded under the output space `LAYER_ACTIVATION`. The same `attributions.pt` and `metadata.json` artifacts are written; the `LayerActivationVisualiser` renders the magnitude summary. `LayerGradCam` and `GuidedGradCam` still produce input-space maps (`IMAGE_SPATIAL_MAP`) and use `CaptumImageVisualiser`.

## `baseline` block

For attribution methods that use a reference input (baseline data), `metadata.json` carries a `baseline`
block documenting the exact reference the explanation was computed against.

```json
"baseline": {
  "kwarg_name": "background_data",
  "mode": "configured",
  "source": "imagenet_samples",
  "n_samples": 50,
  "shape": [50, 3, 224, 224],
  "dtype": "torch.float32",
  "sha256": "…",
  "image_path": "baseline.png"
}
```

- `mode`: how the baseline was obtained: `configured` (resolved from a YAML
  data source), `user_tensor` (passed directly via the Python API), `zero`
  (Captum's implicit all-zeros default), or `input_batch` (SHAP's implicit
  default of using the input batch).
- `source` / `n_samples`: set only for `configured` baselines (the YAML
  provenance).
- `sha256`: content hash of the tensor actually used as the baseline; recorded
  here only, never shown in the report.
- `image_path`: a rendered preview (image modality only), relative to the run
  directory; a single tile for one image, or a capped grid for a multi-image
  baseline. The report shows this image and the descriptor fields, but not the
  hash.
