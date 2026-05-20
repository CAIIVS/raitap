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
{doc}`adding-an-adapter`, {doc}`adding-an-algorithm`, and {doc}`adding-a-module`.

## Assessor hierarchy

Robustness assessors form a three-level hierarchy in
`src/raitap/robustness/assessors/base_assessor.py`:

```text
BaseAssessor                            # root — declares method_kind + budget_kwarg_source
├── EmpiricalAttackAssessor             # you implement generate_adversarial(); framework owns assess()
│   ├── TorchattacksAssessor
│   └── FoolboxAssessor
└── FormalVerificationAssessor          # you implement verify_sample(); framework owns assess()
    └── (alpha-beta-CROWN, auto_LiRPA — follow-up adapters)
```

- **`BaseAssessor`** — root. Declares `method_kind: ClassVar[MethodKind]`,
  `budget_kwarg_source`, and the no-op `check_backend_compat`. Never subclass directly.
- **`EmpiricalAttackAssessor`** — subclasses implement only
  `generate_adversarial(model, inputs, targets, ...) → Tensor`. Batching,
  prediction, verdict computation, distance computation, semantics inference,
  and persistence are owned by this class.
- **`FormalVerificationAssessor`** — subclasses implement
  `verify_sample(model, sample, target, *, budget) → VerificationOutcome`.
  The per-sample loop, runtime tracking, output-bounds stacking, and
  counter-example assembly are owned by this class.

## Class-level attributes the framework reads

Every assessor declares two `ClassVar`s the framework relies on:

- `algorithm_registry: ClassVar[Mapping[str, AssessorSemanticsHints]]` — maps
  algorithm names to their threat model / norm / families. `assessor_semantics`
  uses this to build `RobustnessResult.semantics.budget`, so the reported
  metadata always matches what the adapter actually executed. Passed via the
  `@register_robustness_adapter(algorithm_registry=...)` decorator kwarg.
- `budget_kwarg_source: ClassVar[str]` — `"init_kwargs"` (torchattacks reads
  the budget at construction time) or `"call_kwargs"` (foolbox reads it at
  call time). Defaults to `"init_kwargs"`.

## Visualiser contract

All robustness visualisers implement `BaseRobustnessVisualiser`:

- `visualise(result, *, context, **kwargs) → Figure` — abstract, required.
- `supported_method_kinds: ClassVar[frozenset[MethodKind]]` — empty means
  "all". The factory's `check_assessor_visualiser_compat` enforces this at
  YAML parse time and raises `MethodKindVisualiserIncompatibilityError` on
  mismatch.
- `embeds_clean_input` / `embeds_perturbation_map` — class-level report layout
  hints for empirical visualisers. A visualiser that sets either flag to
  `True` must accept the matching runtime kwarg (`include_clean_input` /
  `include_perturbation_map`) and omit that facet when it is `False`.
- `validate_result(result)` — render-time check that the assessor's
  `method_kind` is in `supported_method_kinds`. Image visualisers additionally
  refuse non-image results via `_require_image_modality`.

The facet flags are consumed only by **compact robustness reporting** to
avoid repeated clean-input or perturbation-map panels across multiple
empirical visualisers. Standalone `RobustnessResult.visualise()` artifacts
remain self-contained and keep the canonical layouts. Formal-verifier
visualisers and custom visualisers that do not opt into these flags need
no changes.

## Typed semantics

`RobustnessResult.semantics` is a typed contract, not a narrative description.
It records method kind, threat model, objective, families, budget, target
classes (for targeted attacks), sample selection, and input metadata.

`MethodKind` distinguishes the two ways to assess robustness:

| Kind | Meaning |
| --- | --- |
| `EMPIRICAL_ATTACK` | Try to find an adversarial example within the budget. A `NOT_ATTACKED` verdict does **not** prove robustness. |
| `FORMAL_VERIFICATION` | Prove (or refute) that no adversarial example exists in the budget. Produces `VERIFIED` / `FALSIFIED` / `UNKNOWN`. |

`RobustnessVerdict` codes the per-sample outcome (encoded as a `long` tensor
in `robustness_data.pt`; the integer mapping is exposed in `metadata.json`
under `verdict_codes`).

`PerturbationBudget` carries `norm`, `epsilon`, `step_size`, and `steps`.
The `norm` value drives `_per_sample_norm` in the empirical pipeline so the
reported `perturbation_distance` always matches the configured threat model.

## Important files

- `contracts.py` — enums and frozen dataclasses for the typed surface.
- `semantics.py` — per-framework registries (`TORCHATTACKS_REGISTRY`,
  `FOOLBOX_REGISTRY`) and the `assessor_semantics(...)` resolver.
- `assessors/base_assessor.py` — the framework-owned `assess()` pipelines for
  both empirical attacks and formal verification.
- `factory.py` — typed config parsing, `_resolve_call_data_sources`, and the
  `RobustnessAssessment` Hydra entry point. Adapter paths are resolved via the
  `@register_robustness_adapter` decorator — no manual path table to maintain.
- `results.py` — `RobustnessResult`, `RobustnessMetrics`, verdict encoding.
- `visualisers/base_visualiser.py` — `BaseRobustnessVisualiser` +
  `MethodKind` compatibility check.
- `visualisers/empirical/` — image-pair and perturbation-heatmap visualisers
  for empirical attacks. The shared `_signed_perturbation_heatmap` helper
  reduces a signed per-channel delta to a 2D scalar map (matplotlib treats
  3-channel arrays as literal RGB and ignores `cmap` / `vmin` / `vmax`, so any
  signed-perturbation render must reduce first).
- `visualisers/formal/` — reserved for the verifier visualiser follow-up
  (verdict badge, certified-bounds plot).

## Runtime flow

1. `RobustnessAssessment(config, name, model, inputs, targets)` creates the
   assessor and its visualisers via the factory.
2. The factory checks method-kind / visualiser compatibility at parse time.
3. `assessor.assess(...)` runs the framework-owned pipeline for the assessor's
   `method_kind` and returns a `RobustnessResult`.
4. `result.write_artifacts()` saves `robustness_data.pt` plus typed metadata.
5. `result.visualise()` iterates configured visualisers, validates each, calls
   `visualise()`, and saves the figures.

**Gotcha**: when the data pipeline returns no labels, the run helper falls
back to `argmax(model(clean_inputs))` so untargeted attacks still have a
well-defined reference (a warning is logged).

## Extending the module

- **New algorithm in an existing adapter (torchattacks, foolbox, ...)** —
  see {doc}`adding-an-algorithm`. For robustness, the `algorithm_registry`
  value is an `AssessorSemanticsHints` (method kind, threat model, objective,
  norm, family tags) from `semantics.py`.
- **New robustness library** — see {doc}`adding-an-adapter`. Pick
  `EmpiricalAttackAssessor` or `FormalVerificationAssessor` as the base,
  decorate with `@register_robustness_adapter(...)`, and set
  `budget_kwarg_source="call_kwargs"` if the library reads the budget at
  call time. Override `check_backend_compat` if the library imposes extra
  backend constraints (e.g. white-box attacks require
  `supports_torch_autograd`).
- **New top-level module** — see {doc}`adding-a-module`.

## Adding a new visualiser

Subclass `BaseRobustnessVisualiser` and decorate with
`@register_robustness_visualiser(...)` (see {doc}`adding-an-adapter` for the
decorator scaffolding). Robustness-specific notes:

- Set `supported_method_kinds` so the factory rejects mismatched assessor
  pairings at parse time.
- For image visualisers, call
  `_require_image_modality(result, type(self).__name__)` inside `visualise()`
  so the `(B, C, H, W)` layout assumption is enforced.
- When rendering signed perturbation deltas, reuse
  `_signed_perturbation_heatmap` from
  `visualisers/empirical/image_pair_visualiser.py` — it preserves the sign
  of the dominant channel without collapsing opposing-sign channels to ~0.
- If your visualiser slots into compact reporting, set
  `embeds_clean_input` / `embeds_perturbation_map` and honour the matching
  runtime kwargs (`include_clean_input` / `include_perturbation_map`).
