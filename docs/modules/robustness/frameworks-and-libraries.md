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

### Formal verification

`FormalVerificationAssessor` provides the framework-owned per-sample loop for
auto_LiRPA / alpha-beta-CROWN style verifiers. Concrete adapters arrive in a
follow-up release; the result shape, factory, pipeline, and reporting already
accommodate the verifier outputs (`VERIFIED` / `FALSIFIED` / `UNKNOWN` verdicts,
counter-examples, per-class output bounds, per-sample runtime).
