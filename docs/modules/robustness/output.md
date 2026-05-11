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

`robustness_data.pt` is a torch checkpoint dict with the following keys (some
are present only for the matching method kind):

| Key | Always present | Description |
| --- | --- | --- |
| `clean_inputs` | yes | Inputs as fed to the model. |
| `targets` | yes | Per-sample reference labels (ground truth or, when no labels were available, model predictions used as the untargeted reference). |
| `clean_predictions` | yes | `argmax(model(clean_inputs))`. |
| `verdicts` | yes | Per-sample integer codes. The mapping is also stored in `metadata.json` under `verdict_codes` (e.g. `attacked → 1`, `not_attacked → 2`, `verified → 3`, `falsified → 4`, `unknown → 5`, `error → 6`). |
| `perturbed_inputs` | empirical / falsified rows | Adversarial example tensor. NaN-padded for verifier rows that did not produce a counter-example. |
| `perturbed_predictions` | empirical / falsified rows | `argmax(model(perturbed_inputs))`; `-1` for verifier rows without a counter-example. |
| `perturbation_distance` | empirical / falsified rows | Per-sample distance under `semantics.budget.norm`. |
| `output_bounds` | formal verification only | `{"lower": ..., "upper": ...}` per-class logit bounds. |
| `runtime_per_sample` | formal verification only | Per-sample verifier runtime, in seconds. |

`metadata.json` carries assessor metadata, aggregate metrics, and the typed
robustness semantics:

- `method_kind` — `empirical_attack` or `formal_verification`.
- `semantics.threat_model` — `white_box`, `black_box_score`, `black_box_decision`.
- `semantics.objective` — `untargeted` or `targeted`.
- `semantics.budget` — norm + epsilon + (optional) step size + steps.
- `semantics.families` — descriptive tags such as `gradient_sign`, `iterative`, `optimization`.
- `metrics` — empirical-only fields (`adversarial_accuracy`, `attack_success_rate`, `mean_distance`, `max_distance`) and formal-only fields (`verified_rate`, `falsified_rate`, `unknown_rate`, `error_rate`, `mean_runtime`); `clean_accuracy` is always populated. Unused fields are omitted rather than set to `null`.
- `kwargs` — RAITAP-owned metadata used for reporting (`sample_names`,
  `show_sample_names`).
- `call_kwargs` — best-effort JSON summary of the library invocation. Tensors
  are summarised (shape, dtype, device) so the metadata stays lightweight.

## Marabou

### Per-logit output bounds (opt-in)

`MarabouAssessor` can populate `RobustnessResult.output_bounds` with certified
per-class logit ranges for each VERIFIED sample. Enable via the constructor:

| Kwarg | Default | Meaning |
|---|---|---|
| `compute_output_bounds` | `False` | Master switch. When `True`, run bisection-via-SAT after each VERIFIED verdict. |
| `bound_search_range` | `1e3` | Initial probe window `[-range, +range]` per output variable. |
| `bound_tolerance` | `1e-2` | Stop bisection when the certified interval narrows below this. |

**Runtime cost.** Marabou exposes no native min/max objective; bounds are
extracted by binary search on `setUpperBound` / `setLowerBound` of each output
variable. Per verified sample, the assessor runs up to

`2 × K × (⌈log₂(2 × bound_search_range / bound_tolerance)⌉ + 2)`

extra Marabou solves. With `K=10` classes and defaults
(`bound_search_range=1e3`, `bound_tolerance=1e-2`) that is up to ~400
additional solves per sample. FALSIFIED / UNKNOWN / ERROR samples are
skipped entirely (their rows in the stacked bounds tensor are NaN-padded);
if a bisection probe itself times out the loop breaks early with a
conservative bound, so real-world solve counts are typically lower.

Inconclusive verdicts during bisection (TIMEOUT / UNKNOWN) break the search
conservatively: the returned bound is the loosest still-certified value, never
a falsely tight one. If *every* probe for a given class/mode is inconclusive
the assessor emits a `WARNING` log so users can spot vacuous bounds (the
returned value is then just `±bound_search_range`).

The PDF report's Robustness section gains rows `logit_{k}_lower_mean`,
`logit_{k}_upper_mean` (averaged across samples that have bounds) and
`output_bounds_samples` (count of verified samples with bounds /
total samples). Visualisation of these bounds is tracked separately in
issue #141.
