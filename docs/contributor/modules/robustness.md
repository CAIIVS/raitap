---
title: "Contributing to the robustness module"
description: "Robustness-module-specific architecture: assessor hierarchy, typed semantics contract, and visualiser contract."
myst:
  html_meta:
    "description": "Robustness-module-specific architecture: assessor hierarchy, typed semantics contract, and visualiser contract."
---

# Contributing to the robustness module

This page covers what's specific to the robustness module. For the generic
"how do I plug in a library / algorithm / new module?" walkthroughs see
{doc}`../adding/adding-an-adapter`, {doc}`../adding/adding-an-algorithm`, and {doc}`../adding/adding-a-module`.

## Assessor hierarchy

Robustness assessors form a three-level hierarchy in
`src/raitap/robustness/assessors/base_assessor.py`:

```text
BaseAssessor                            # root: declares assessment_kind + budget_kwarg_source
├── EmpiricalAttackAssessor             # you implement _default_invoke(ctx); framework owns generate_adversarial() + assess()
│   ├── TorchattacksAssessor
│   └── FoolboxAssessor
├── FormalVerificationAssessor          # you implement verify_sample(); framework owns assess()
│   ├── MarabouAssessor                 # complete SMT (ONNX MLPs)
│   └── AutoLiRPAAssessor               # sound+incomplete bound propagation (CROWN/IBP)
└── StatisticalSamplingAssessor         # you implement apply_perturbation(image); framework owns assess()
    └── ImageCorruptionsAssessor
```

- **`BaseAssessor`**: root. Declares `assessment_kind: ClassVar[AssessmentKind]`
  and `budget_kwarg_source`. Backend gating is inherited from `AdapterMixin`
  (`check_backend_compat`), not a no-op on this class. Never subclass directly.
- **`EmpiricalAttackAssessor`**: subclasses implement only
  `_default_invoke(self, ctx: AttackInvokeCtx) -> Tensor`. `generate_adversarial`
  is the framework-owned dispatcher: it resolves the per-entry invoker (or falls
  back to `_default_invoke`) then delegates. Batching, prediction, verdict
  computation, distance computation, semantics inference, and persistence are
  owned by this class.
- **`FormalVerificationAssessor`**: subclasses implement
  `verify_sample(model, sample, target, *, budget) -> VerificationOutcome`.
  The per-sample loop, runtime tracking, output-bounds stacking, and
  counter-example assembly are owned by this class.
- **`StatisticalSamplingAssessor`**: subclasses implement only
  `apply_perturbation(image: np.ndarray) -> np.ndarray` on a single HWC
  uint8 image. The framework owns batching, forward passes, verdict assignment,
  CI computation, and persistence.

## Class-level attributes the framework reads

Every assessor declares two `ClassVar`s the framework relies on:

- `algorithm_registry: ClassVar[Mapping[str, AssessorAlgorithmSpec]]`: maps
  algorithm names to their threat model / norm / families / `stochastic` flag.
  `assessor_semantics` uses this to build `RobustnessResult.semantics`, so the
  reported metadata always matches what the adapter actually executed. The
  `stochastic: bool` hint is declared explicitly per algorithm (e.g. `True` for
  PGD random-start and statistical-sampling corruptions, `False` for FGSM / CW);
  it flows onto `semantics.stochastic` and drives the reproducibility caveat.
  Passed via the `@adapters.robustness(algorithm_registry=...)` decorator kwarg.
- `budget_kwarg_source: ClassVar[str]`: `"init_kwargs"` (torchattacks reads
  the budget at construction time) or `"call_kwargs"` (foolbox reads it at
  call time). Defaults to `"init_kwargs"`.

## Visualiser contract

All robustness visualisers implement `BaseRobustnessVisualiser`:

- `visualise(result, *, context, **kwargs) -> Figure`: abstract, required.
- `supported_assessment_kinds: ClassVar[frozenset[AssessmentKind]]`: empty means
  "all". The factory's `check_assessor_visualiser_compat` enforces this at
  YAML parse time and raises `AssessmentKindVisualiserIncompatibilityError` on
  mismatch.
- `embeds_clean_input` / `embeds_perturbation_map`: class-level report layout
  hints for empirical visualisers. A visualiser that sets either flag to
  `True` must accept the matching runtime kwarg (`include_clean_input` /
  `include_perturbation_map`) and omit that facet when it is `False`.
- `validate_result(result)`: render-time check that the assessor's
  `assessment_kind` is in `supported_assessment_kinds`. Image visualisers additionally
  refuse non-image results via `_require_image_modality`.

The facet flags are consumed only by **compact robustness reporting** to
avoid repeated clean-input or perturbation-map panels across multiple
empirical visualisers. Standalone `RobustnessResult.visualise()` artifacts
remain self-contained and keep the canonical layouts. Formal-verifier
visualisers and custom visualisers that do not opt into these flags need
no changes.

## Typed semantics

`RobustnessResult.semantics` is a typed contract, not a narrative description.
It records assessment kind, threat model, objective, families, perturbation, target
classes (for targeted attacks), sample selection, and input metadata.

`AssessmentKind` is the procedure-level taxonomy (Level 1). Each kind belongs
to exactly one `RobustnessCase` (Level 2) derived via `case_for(kind)` (never
stored independently, but surfaced as the `case` key in `metadata.json`).

| Kind | Case | Meaning |
| --- | --- | --- |
| `EMPIRICAL_ATTACK` | `worst_case` | Try to find an adversarial example within the budget. `ATTACK_FAILED` does **not** prove robustness. |
| `FORMAL_VERIFICATION` | `worst_case` | Prove (or refute) that no adversarial example exists in the budget. Produces `VERIFIED` / `FALSIFIED` / `UNKNOWN`. |
| `STATISTICAL_SAMPLING` | `average_case` | Measure accuracy under a perturbation distribution. Produces `CORRECT_UNDER_PERTURBATION` / `MISCLASSIFIED_UNDER_PERTURBATION`. |

`RobustnessVerdict` codes the per-sample outcome (encoded as a `long` tensor
in `robustness_data.pt`; the integer mapping is exposed in `metadata.json`
under `verdict_codes`). Empirical assessors emit `ATTACK_SUCCEEDED` /
`ATTACK_FAILED`; formal assessors emit `VERIFIED` / `FALSIFIED` / `UNKNOWN` /
`ERROR`; statistical-sampling assessors emit `CORRECT_UNDER_PERTURBATION` /
`MISCLASSIFIED_UNDER_PERTURBATION`.

`RobustnessSemantics.perturbation` is a `PerturbationRegion` base type. Worst-case
assessors use `PerturbationBudget` (carries `norm`, `epsilon`, `step_size`,
`steps`; the `norm` drives `_per_sample_norm` so reported `perturbation_distance`
always matches the configured threat model). Average-case assessors use
`PerturbationDistribution` (carries `corruption_name` and `severity`).

`ThreatModel.NOT_APPLICABLE` is used by statistical-sampling assessors where
there is no adversary; empirical and formal assessors use
`WHITE_BOX`, `BLACK_BOX_SCORE`, or `BLACK_BOX_DECISION`.

## Important files

- `contracts.py`: enums and frozen dataclasses for the typed surface.
- `semantics.py`: per-framework registries (`TORCHATTACKS_REGISTRY`,
  `FOOLBOX_REGISTRY`) and the `assessor_semantics(...)` resolver.
- `assessors/base_assessor.py`: the framework-owned `assess()` pipelines for
  both empirical attacks and formal verification.
- `factory.py`: typed config parsing and the `RobustnessAssessment` Hydra entry
  point (data-source resolution is shared via `configs/adapter_factory`). Adapter
  paths are resolved via the `@adapters.robustness` decorator (no manual path
  table to maintain).
- `results.py`: `RobustnessResult`, `RobustnessMetrics`, verdict encoding.
- `visualisers/base_visualiser.py`: `BaseRobustnessVisualiser` +
  `AssessmentKind` compatibility check.
- `visualisers/empirical/`: image-pair and perturbation-heatmap visualisers
  for empirical attacks. The shared `_signed_perturbation_heatmap` helper
  reduces a signed per-channel delta to a 2D scalar map (matplotlib treats
  3-channel arrays as literal RGB and ignores `cmap` / `vmin` / `vmax`, so any
  signed-perturbation render must reduce first).
- `visualisers/formal/`: reserved for the verifier visualiser follow-up
  (verdict badge, certified-bounds plot).
- `assessors/imagecorruptions_assessor.py`: `ImageCorruptionsAssessor`; wraps
  19 ImageNet-C corruptions (15 common + 4 holdout) via `imagecorruptions`.
- `assessors/auto_lirpa_assessor.py`: `AutoLiRPAAssessor`; sound+incomplete
  bound-propagation verifier (CROWN / IBP) via `auto_LiRPA`. The algorithm key
  is the single source of truth for both the bound method and the norm
  (`crown`/`ibp`/`crown-ibp` -> L-inf, `crown-l2` -> L2); `verify_sample` reads the
  norm off `budget.norm` and maps it to `PerturbationLpNorm`. Its algorithms
  carry `requires={Capability.AUTOGRAD}`, so the inherited
  `check_backend_compat` rejects ONNX/forward-only backends automatically.
  The class overrides `check_backend_compat` only to call `super()` first and
  then warn on Intel XPU.
- `visualisers/average_case/corruption_accuracy_visualiser.py`:
  `CorruptionAccuracyVisualiser`; renders clean vs corrupted accuracy bars with
  a CI whisker.

## Runtime flow

1. `RobustnessAssessment(config, name, model, inputs, targets)` creates the
   assessor and its visualisers via the factory.
2. The factory checks assessment-kind / visualiser compatibility at parse time.
3. `assessor.assess(...)` runs the framework-owned pipeline for the assessor's
   `assessment_kind` and returns a `RobustnessResult`.
4. `result.write_artifacts()` saves `robustness_data.pt` plus typed metadata.
5. `result.visualise()` iterates configured visualisers, validates each, calls
   `visualise()`, and saves the figures.

**Gotcha**: when the data pipeline returns no labels, the run helper falls
back to `argmax(model(clean_inputs))` so untargeted attacks still have a
well-defined reference (a warning is logged).

## Invoker seam

`AssessorAlgorithmSpec.invoker` overrides the adapter's default
`_default_invoke(ctx)` construct-and-call path for one specific registry entry.
`None` (the default, ~95% of entries) means the adapter's own `_default_invoke`
runs. The field carries any callable matching the generic `Invoker` Protocol in
`src/raitap/_adapters.py`:

```python
class Invoker(Protocol[CtxT, ResultT]):
    def __call__(self, ctx: CtxT, /) -> ResultT: ...
```

For robustness, `CtxT` is `AttackInvokeCtx` (defined in `base_assessor.py`).
The context dataclass carries the assessor instance so a custom invoker can
reuse every shared helper (`_rethrow`, `_prepare_inputs_for_forward`,
`_maybe_set_targeted`, `_extract_scalar_eps`, `_build_criterion`,
`_last_success`) without reimplementing them.

**Worked example: `DatasetAttack`.** foolbox's `DatasetAttack` has a
two-stage lifecycle: you must call `.feed(fmodel, inputs)` to populate the
sample pool before calling the attack. The uniform `_default_invoke` path
(construct, then call) cannot express this. The solution is a module-level
function in `foolbox_assessor.py`:

```python
def _dataset_attack_invoker(ctx: AttackInvokeCtx) -> torch.Tensor:
    ...
    attack.feed(fmodel, inputs_dev)   # pool population
    ...
    _raw, clipped, success = attack(fmodel, inputs_dev, targets_dev, epsilons=eps)
    return clipped.detach()
```

The registry entry passes it via:

```python
"DatasetAttack": _hint(..., invoker=_dataset_attack_invoker),
```

The invoker pattern is also used by `JSMA` in `torchattacks_assessor.py` to
guard against the hardcoded 10-class assumption before delegating back to
`_default_invoke`.

See {doc}`../adding/adding-an-algorithm` for the cross-family picture
(including the transparency SHAP invokers).

## Extending the module

- **New algorithm in an existing adapter (torchattacks, foolbox, ...)**:
  see {doc}`../adding/adding-an-algorithm`. For robustness, the `algorithm_registry`
  value is an `AssessorAlgorithmSpec` (assessment kind, threat model, objective,
  norm, family tags) from `semantics.py`.
- **New robustness library**: see {doc}`../adding/adding-an-adapter`. Pick
  `EmpiricalAttackAssessor`, `FormalVerificationAssessor`, or
  `StatisticalSamplingAssessor` as the base, decorate with
  `@adapters.robustness(...)`, and set `budget_kwarg_source="call_kwargs"` if
  the library reads the budget at call time. Backend gating is automatic: the
  gate inherited from `AdapterMixin` evaluates whether
  `algorithm.requires <= backend.provides` and raises
  `BackendIncompatibilityError` on mismatch. Set
  `requires={Capability.AUTOGRAD}` on `AssessorAlgorithmSpec`
  entries for algorithms that need autograd (e.g. white-box empirical attacks).
  Statistical-sampling adapters implement `apply_perturbation(image)` only;
  their algorithms carry empty `requires` so the gate always passes.
  Do NOT override `check_backend_compat` for normal adapters. The only valid
  override cases are: adding a per-call structural check (Marabou pattern) or
  extending with a non-capability warning after calling `super()` (auto-LiRPA
  pattern).
- **New top-level module**: see {doc}`../adding/adding-a-module`.

## The auto-LiRPA dependency and the torch 2.8 project pin

`auto_LiRPA` is the one robustness dependency that is **git-only**: its last PyPI
release (0.3, Sept 2022) supports only `torch<1.13`; torch-2.x support lives on
GitHub master. Two consequences contributors should know:

- **PyPI-legal declaration.** The `auto-lirpa` extra lists the requirement by
  bare name (`auto-LiRPA`); the git URL lives in `[tool.uv.sources]`. uv sources
  are not written into wheel metadata, so raitap's published wheel stays
  installable from PyPI. A direct `@ git+https://...` reference in
  `[project.optional-dependencies]` would land in `Requires-Dist` and make
  `twine upload` reject the wheel. Never inline the URL into the extra.
- **Project-wide torch 2.8 pin.** auto-LiRPA master pins `torch>=2.0.0,<2.9.0`,
  so all torch/onnx extras floor at `torch>=2.8.0,<2.9.0` (down from the original
  `>=2.10.0` scaffolding default, because no code used a torch 2.9/2.10-only API).
  This keeps a single coherent environment instead of a forked lockfile, at the cost
  of the 2.9-2.12 line. xpu/cpu/cuda wheels exist for cp311/312/313, covering
  `requires-python >=3.11,<3.14`. `uv lock` resolves the git build on Linux/CI;
  the upstream `setup.py` reads a file without an explicit UTF-8 encoding, so the
  build (and thus `uv sync --extra auto-lirpa`) fails on Windows. Verify lock
  resolution on CI, not a Windows checkout.

## Adding a new visualiser

Subclass `BaseRobustnessVisualiser` and decorate with
`@visualisers.robustness(...)` (see {doc}`../adding/adding-an-adapter` for the
decorator scaffolding). Robustness-specific notes:

- Set `supported_assessment_kinds` so the factory rejects mismatched assessor
  pairings at parse time.
- For image visualisers, call
  `_require_image_modality(result, type(self).__name__)` inside `visualise()`
  so the `(B, C, H, W)` layout assumption is enforced.
- When rendering signed perturbation deltas, reuse
  `_signed_perturbation_heatmap` from
  `visualisers/empirical/image_pair_visualiser.py`. It preserves the sign
  of the dominant channel without collapsing opposing-sign channels to ~0.
- If your visualiser slots into compact reporting, set
  `embeds_clean_input` / `embeds_perturbation_map` and honour the matching
  runtime kwargs (`include_clean_input` / `include_perturbation_map`).
- Set `report_figure_scope` to declare where the report places the figure:
  `PER_SAMPLE` (default, one figure per input, e.g. image pairs) or `ASSESSOR`
  (one figure summarising the whole assessment, e.g. accuracy bars, verdict
  summaries, output-bound plots). The reporting layer reads it to pick the layout
  slot, so an assessor-level visualiser renders correctly with no reporting-layer
  changes. (Consumed by the HTML report today; PDF is a tracked follow-up.)
