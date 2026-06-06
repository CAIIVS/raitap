---
title: "Output"
description: "If an assessor uses more than one visualiser, RAITAP writes one PNG per visualiser using the pattern <VisualiserClassName>_<index>.png. These standalone PNGs remain self-contained even when the PDF report uses a compact robustness layout."
myst:
  html_meta:
    "description": "If an assessor uses more than one visualiser, RAITAP writes one PNG per visualiser using the pattern <VisualiserClassName><index>.png. These standalone PNGs remain self-contained even when the PDF report uses a compact robustness layout. Th"
---

# Output

```text
└── robustness/
    ├── pgd/                                # Output for the `pgd` assessor
    │   ├── robustness_data.pt              # Clean / perturbed inputs, predictions, verdicts, distances
    │   ├── ImagePairVisualiser_0.png       # Visualisation written by the first visualiser
    │   └── metadata.json                   # Assessor metadata + serialised semantics
    └── fgsm/                               # One subdirectory per named assessor
        ├── robustness_data.pt
        ├── ImagePairVisualiser_0.png
        └── metadata.json
```

If an assessor uses more than one visualiser, RAITAP writes one PNG per
visualiser using the pattern `<VisualiserClassName>_<index>.png`.
These standalone PNGs remain self-contained even when the PDF report uses a
compact robustness layout. The `metadata.json` `visualisers` list always
references the canonical persisted layouts, not report-only variants.

`robustness_data.pt` is a torch checkpoint dict with the following keys (some
are present only for the matching assessment kind):

| Key | Always present | Description |
| --- | --- | --- |
| `clean_inputs` | yes | Inputs as fed to the model. |
| `targets` | yes | Per-sample reference labels (ground truth or, when no labels were available, model predictions used as the untargeted reference). |
| `clean_predictions` | yes | `argmax(model(clean_inputs))`. |
| `verdicts` | yes | Per-sample integer codes. The mapping is also stored in `metadata.json` under `verdict_codes` (e.g. `attack_succeeded → 1`, `attack_failed → 2`, `verified → 3`, `falsified → 4`, `unknown → 5`, `error → 6`, `correct_under_perturbation → 7`, `misclassified_under_perturbation → 8`). |
| `perturbed_inputs` | empirical / falsified rows / statistical sampling | Adversarial example tensor (empirical), counter-example NaN-padded for verifier rows without one (formal), or the corrupted inputs (statistical sampling). |
| `perturbed_predictions` | empirical / falsified rows / statistical sampling | `argmax(model(perturbed_inputs))`; `-1` for verifier rows without a counter-example. |
| `perturbation_distance` | empirical / falsified rows | Per-sample distance under `semantics.perturbation.norm` (worst-case only; absent for statistical sampling). |
| `output_bounds` | formal verification only | `{"lower": ..., "upper": ...}` per-class logit bounds. |
| `runtime_per_sample` | formal verification only | Per-sample verifier runtime, in seconds. |

`metadata.json` carries assessor metadata, aggregate metrics, and the typed
robustness semantics:

- `assessment_kind` — `empirical_attack`, `formal_verification`, or `statistical_sampling`.
- `case` — `worst_case` (empirical / formal) or `average_case` (statistical sampling); derived from `assessment_kind`, never stored independently.
- `semantics.threat_model` — `white_box`, `black_box_score`, `black_box_decision`, or `not_applicable` (statistical sampling).
- `semantics.objective` — `untargeted` or `targeted`.
- `semantics.perturbation` — worst-case assessors: norm + epsilon + (optional) step size + steps; average-case assessors: corruption name + severity.
- `semantics.families` — descriptive tags such as `gradient_sign`, `iterative`, `optimization`, or `common_corruption`.
- `semantics.stochastic` — `true` when the algorithm is RNG-dependent (random start, sampling, statistical sampling), so the result is not bit-reproducible unless seeds are pinned. Drives the run-level reproducibility caveat.
- `metrics` — empirical-only fields (`adversarial_accuracy`, `attack_success_rate`, `mean_distance`, `max_distance`); formal-only fields (`verified_rate`, `falsified_rate`, `unknown_rate`, `error_rate`, `mean_runtime`); statistical-sampling-only fields (`corrupted_accuracy`, `accuracy_ci_low`, `accuracy_ci_high`, `n_samples`, `n_correct`); `clean_accuracy` is always populated. Unused fields are omitted rather than set to `null`.
- `kwargs` — RAITAP-owned metadata used for reporting (`sample_names`,
  `show_sample_names`).
- `call_kwargs` — best-effort JSON summary of the library invocation. Tensors
  are summarised (shape, dtype, device) so the metadata stays lightweight.

## `output_bounds` (formal verification)

When a formal-verification assessor populates per-logit certified ranges,
`RobustnessResult.output_bounds` is a dict with `"lower"` and `"upper"`
tensors of shape `(N, K)` (NaN-padded for samples without bounds). The PDF
report adds rows `output_bounds_samples`, `logit_{k}_lower_mean`, and
`logit_{k}_upper_mean` to the Robustness section. See
{doc}`frameworks-and-libraries` for the assessor-side knobs that produce
these bounds, and {doc}`visualisers` for the formal-verification visualisers
that render them.
