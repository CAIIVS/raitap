---
title: "Supported libraries"
description: "Assessors support three config buckets:"
myst:
  html_meta:
    "description": "Assessors support three config buckets:"
---

# Supported libraries

## `constructor`, `call`, and `raitap` keys

Assessors support three config buckets:

- `constructor`: kwargs for the assessor constructor or underlying library object
- `call`: verbatim library kwargs for the per-call attack invocation
- `raitap`: RAITAP-owned runtime options such as batching, progress bars, and
  sample-name metadata

Visualisers continue to support `constructor` and `call` only.

This keeps the boundary clear for users: `call` is what the library sees at
attack time, while `raitap` is what RAITAP itself consumes.

The perturbation budget keys (`eps`, `alpha`, `steps`) live in only **one** of
`constructor:` and `call:` per framework; the other source is ignored by the
adapter. RAITAP picks the authoritative side automatically:

| Adapter | Budget block | Why |
| --- | --- | --- |
| `TorchattacksAssessor` | `constructor:` | The adapter does `attack_class(model, **constructor)` once and never forwards per-call budget kwargs. |
| `FoolboxAssessor` | `call:` | Foolbox attacks read `epsilons=...` at `attack(fmodel, inputs, targets, epsilons=...)`. |

Putting budget kwargs in the wrong block emits a warning so the misconfiguration
is visible in the run log.

## Typed semantics and visualiser compatibility

RAITAP uses typed `MethodKind`, `ThreatModel`, `Objective`, and
`PerturbationBudget` semantics to validate visualisers against the result
they receive. In short:

- assessors produce typed `RobustnessResult.semantics`
- visualisers declare which `MethodKind` they can render via the
  `supported_method_kinds: ClassVar[frozenset[MethodKind]]` attribute
- the factory rejects incompatible pairings at YAML parse time
  (`MethodKindVisualiserIncompatibilityError`)
- image visualisers additionally refuse non-image results
  (`input_spec.kind != IMAGE`)

| Visualiser | Supports | Notes |
| --- | --- | --- |
| `ImagePairVisualiser` | `EMPIRICAL_ATTACK` | Renders N rows by 3 columns: clean, perturbed, signed perturbation heatmap. Rejects tabular / time-series / token results. |
| `PerturbationHeatmapVisualiser` | `EMPIRICAL_ATTACK` | Per-sample diverging heatmap of the perturbation. Default channel reduction is `signed_dominant` (preserves sign without cancelling opposing channels). Other modes: `mean`, `mean_abs`, `max_abs`. |
| `VerdictSummaryVisualiser` | `FORMAL_VERIFICATION` | Two-panel summary: verdict-count bar chart plus a runtime histogram per verified sample. |
| `OutputBoundsCohortVisualiser` | `FORMAL_VERIFICATION` | Boxplot of certified per-class output-bound widths (`upper - lower`) across the verified batch. Constructor kwargs: `whis`, `show_outliers`. Renders a placeholder figure when `result.output_bounds is None` or all rows are NaN. |
| `OutputBoundsPinnedVisualiser` | `FORMAL_VERIFICATION` | Per-sample plot of `[lower_k, upper_k]` certified intervals for each output class with the target class highlighted. Constructor kwargs: `max_samples`, `target_color`, `bar_color`, `sample_indices`. Falls back to a placeholder when bounds are absent. |
| `OutputBoundsWidthHeatmapVisualiser` | `FORMAL_VERIFICATION` | Heatmap of certified per-class output-bound widths (`upper - lower`) across the verified batch (rows = samples, columns = classes). Constructor kwargs: `cmap`, `max_samples`, `figsize`. Renders a placeholder figure when `result.output_bounds is None` or every row is NaN. |
| `OutputBoundsMarginHeatmapVisualiser` | `FORMAL_VERIFICATION` | Heatmap of signed per-class margins relative to the target class's lower bound (rows = samples, columns = classes; target cell masked). Constructor kwargs: `cmap`, `max_samples`, `figsize`. Falls back to a placeholder when bounds or targets are absent. |

Empirical image visualisers declare whether they embed a clean-input panel or a
perturbation-map panel by default. Compact reporting uses those declarations to
choose one canonical owner per facet and ask non-owners to omit repeated panels.
The runtime kwargs are `include_clean_input` and `include_perturbation_map`.
They affect report-only renders; persisted visualiser PNGs remain self-contained.
Verifier visualisers keep the default facet flags (`False`) and do not need to
accept these kwargs unless they explicitly opt into the contract.

Contributor-facing details about the assessor / visualiser internals are in
{doc}`../../contributor/robustness`.

## Assessor libraries

### Torchattacks

`TorchattacksAssessor` wraps every attack class in `torchattacks` via dynamic
loading; the YAML `algorithm:` field names the class. White-box attacks require
a torch backend with autograd (the adapter rejects ONNX backends with
`AssessorBackendIncompatibilityError`). Inputs are made contiguous before the
call so attacks that internally `view(...)` (e.g. `PGDL2`, `CW`, `DeepFool`,
`Square`) work on RAITAP's loader output (which produces non-contiguous NCHW
tensors via HWC→CHW transpose).

A non-exhaustive sample of supported algorithms:

| Algorithm | Threat model | Norm | Notes |
| --- | --- | --- | --- |
| `FGSM` | white-box | L∞ | Single-step gradient sign. CPU-friendly. |
| `BIM` | white-box | L∞ | Iterative FGSM. |
| `PGD` | white-box | L∞ | Projected gradient descent. |
| `PGDL2` | white-box | L2 | L2 variant of PGD. |
| `CW` | white-box | L2 | Carlini-Wagner optimisation attack. |
| `DeepFool` | white-box | L2 | Iterative linearisation. |
| `MIFGSM` | white-box | L∞ | Momentum-iterative FGSM. |
| `AutoAttack` | white-box | L∞ | Ensemble of attacks; expensive. |
| `Square` | black-box (score) | L∞ | Score-based query attack. |
| `OnePixel` | black-box (score) | L0 | Differential-evolution single-pixel attack. |

### Foolbox

`FoolboxAssessor` wraps `foolbox.attacks.<algorithm>` against a
`foolbox.PyTorchModel(model, bounds=..., preprocessing=...)`. Bounds default to
`(0.0, 1.0)`, matching RAITAP's loader. The adapter accepts only **scalar**
`eps` / `epsilons`; multi-epsilon sweeps would change the result tensor shape
across configurations and break the uniform `RobustnessResult` contract — they
are intentionally out of scope for the current adapter.

| Algorithm | Threat model | Norm |
| --- | --- | --- |
| `LinfPGD` | white-box | L∞ |
| `L2PGD` | white-box | L2 |
| `LinfFastGradientAttack` | white-box | L∞ |
| `L2FastGradientAttack` | white-box | L2 |
| `L2CarliniWagnerAttack` | white-box | L2 |
| `L2DeepFoolAttack` | white-box | L2 |
| `BoundaryAttack` | black-box (decision) | L2 |

### Marabou

`MarabouAssessor` wraps `maraboupy>=2.0` to provide SAT/UNSAT-based formal
verification for L∞ box perturbations over static-shape ONNX MLPs. Verdicts
land in `RobustnessResult.verdicts` (`VERIFIED` / `FALSIFIED` / `UNKNOWN` /
`ERROR`) and counter-examples in `perturbed_inputs`.

#### Algorithms

| `algorithm` | Property |
| --- | --- |
| `linf-box` | Per-input box `[x_i - eps, x_i + eps]` plus an output disjunction asserting "any non-target class dominates the target". UNSAT → VERIFIED, SAT → FALSIFIED with reconstructed counter-example. |

#### Per-logit output bounds (opt-in)

`MarabouAssessor` can additionally certify per-class logit ranges for each
VERIFIED sample, populating `RobustnessResult.output_bounds`.

| Kwarg | Default | Meaning |
| --- | --- | --- |
| `compute_output_bounds` | `False` | Enable bisection-via-SAT bound extraction after each VERIFIED verdict. |
| `bound_search_range` | `1e3` | Initial probe window `[-range, +range]` per output variable. |
| `bound_tolerance` | `1e-2` | Stop bisection when the certified interval narrows below this. |

Marabou exposes no native min/max objective, so bounds are extracted by
binary search on `setUpperBound` / `setLowerBound` of each output variable.
Per verified sample, the assessor runs up to
`2 × K × (⌈log₂(2 × bound_search_range / bound_tolerance)⌉ + 2)` extra
Marabou solves — for `K=10` classes with defaults that is up to ~400 extra
solves per sample. FALSIFIED / UNKNOWN / ERROR samples are skipped (their
rows in the stacked bounds tensor are NaN-padded). Inconclusive verdicts
during bisection (TIMEOUT / UNKNOWN) break the search conservatively; the
returned bound is the loosest still-certified value, never a falsely tight
one. If every probe for a given class/mode is inconclusive the assessor
emits a `WARNING` log so vacuous bounds are obvious.
