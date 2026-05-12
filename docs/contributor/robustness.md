# Contributing to the robustness module

This page describes the internal robustness architecture and how to extend it
with new algorithms, frameworks, and visualisers.

## Overview

The robustness module wraps adversarial-attack and formal-verification
libraries behind a unified interface driven by Hydra `_target_` instantiation.
Assessors produce a `RobustnessResult` with typed semantics; visualisers
validate those semantics before rendering.

Assessors form a three-level hierarchy (see
`src/raitap/robustness/assessors/base_assessor.py`):

```text
BaseAssessor                            # root — declares method_kind + budget_kwarg_source
├── EmpiricalAttackAssessor             # you implement generate_adversarial(); framework owns assess()
│   ├── TorchattacksAssessor
│   └── FoolboxAssessor
└── FormalVerificationAssessor          # you implement verify_sample(); framework owns assess()
    └── (alpha-beta-CROWN, auto_LiRPA — follow-up adapters)
```

- **`BaseAssessor`** — root. Declares `method_kind: ClassVar[MethodKind]`,
  `threat_model_default`, `objective_default`, `budget_kwarg_source` (which
  YAML block the underlying library actually consumes for the perturbation
  budget), and the no-op `check_backend_compat`. Never subclass directly.
- **`EmpiricalAttackAssessor`** — extend this when the framework should manage
  the full `assess` pipeline. Subclasses implement only
  `generate_adversarial(model, inputs, targets, ...) → Tensor`; batching,
  prediction, verdict computation, distance computation, semantics inference,
  and persistence are handled by this class.
- **`FormalVerificationAssessor`** — extend this for verifiers. Subclasses
  implement `verify_sample(model, sample, target, *, budget) →
  VerificationOutcome`; the per-sample loop, runtime tracking, output-bounds
  stacking, and counter-example assembly are handled by this class.

Each adapter declares two class-level attributes the framework reads:

- `algorithm_registry: ClassVar[Mapping[str, AssessorSemanticsHints]]` — maps
  algorithm names to their threat model / norm / families. Used by
  `assessor_semantics` to build `RobustnessResult.semantics.budget` so the
  reported metadata always matches what the adapter executed.
- `budget_kwarg_source: ClassVar[str]` — `"init_kwargs"` or `"call_kwargs"`,
  depending on whether the underlying library reads the budget at
  construction time (torchattacks) or at call time (foolbox).

All visualisers implement `BaseRobustnessVisualiser`, which defines:

- `visualise(result, *, context, **kwargs) → Figure` (abstract, required)
- `supported_method_kinds: ClassVar[frozenset[MethodKind]]` — empty means "all".
  The factory's `check_assessor_visualiser_compat` enforces this at YAML parse
  time so configuration errors fail fast (raises
  `MethodKindVisualiserIncompatibilityError`).
- `embeds_clean_input` / `embeds_perturbation_map` — class-level report layout
  hints for empirical visualisers. A visualiser that sets either flag to `True`
  must accept the matching runtime kwarg (`include_clean_input` /
  `include_perturbation_map`) and omit that facet when it is `False`.
- `validate_result(result)` — render-time check that the assessor's
  `method_kind` is in `supported_method_kinds`. Image visualisers additionally
  refuse non-image results via `_require_image_modality`.

The facet flags are used only by compact robustness reporting to avoid repeated
clean-input or perturbation-map panels across multiple empirical visualisers.
Standalone `RobustnessResult.visualise()` artifacts remain self-contained and
continue to use the canonical layouts. Formal verifier visualisers and custom
visualisers that do not opt into these flags need no changes.

### Typed semantics contract

`RobustnessResult.semantics` describes the executed assessment. It is a typed
contract, not a narrative description. The contract records the method kind,
threat model, objective, families, budget, target classes (for targeted
attacks), sample selection, and input metadata.

`MethodKind` distinguishes the two ways to assess robustness:

| Kind | Meaning |
| --- | --- |
| `EMPIRICAL_ATTACK` | Try to find an adversarial example within the budget. A `NOT_ATTACKED` verdict does **not** prove robustness. |
| `FORMAL_VERIFICATION` | Prove (or refute) that no adversarial example exists in the budget. Produces `VERIFIED` / `FALSIFIED` / `UNKNOWN`. |

`RobustnessVerdict` codes the per-sample outcome (encoded as a `long` tensor in
`robustness_data.pt`; the integer mapping is exposed in `metadata.json` under
`verdict_codes`).

`PerturbationBudget` carries `norm`, `epsilon`, `step_size`, and `steps`. The
`norm` value drives `_per_sample_norm` in the empirical pipeline so reported
`perturbation_distance` always matches the configured threat model.

## Important files

- `contracts.py` — enums and frozen dataclasses for the typed surface.
- `semantics.py` — per-framework registries (`TORCHATTACKS_REGISTRY`,
  `FOOLBOX_REGISTRY`) and the `assessor_semantics(...)` resolver.
- `assessors/base_assessor.py` — the framework-owned `assess()` pipelines for
  both empirical attacks and formal verification.
- `factory.py` — typed config parsing, `_resolve_call_data_sources`, and the
  `RobustnessAssessment` Hydra entry point. Bare class names are resolved to
  `raitap.robustness.assessors.*` and `raitap.robustness.visualisers.*` paths.
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

Robustness runs after the transparency loop in `src/raitap/run/pipeline.py`.
For each configured assessor:

1. `RobustnessAssessment(config, name, model, inputs, targets)` creates the
   assessor and visualisers via the factory.
2. The factory enforces method-kind / visualiser compatibility at parse time.
3. `assessor.assess(...)` runs the framework-owned pipeline for the assessor's
   `method_kind` and returns a `RobustnessResult`.
4. `result.write_artifacts()` saves `robustness_data.pt` plus typed metadata.
5. `result.visualise()` iterates configured visualisers, validates each, calls
   `visualise()`, and saves the figures.

When the data pipeline returns no labels, the run helper falls back to
`argmax(model(clean_inputs))` so untargeted attacks still have a well-defined
reference (a warning is logged).

## Adding a new algorithm

Both adapters dispatch to algorithms dynamically via `getattr`. Adding a new
algorithm typically only requires extending the corresponding registry in
`semantics.py` so semantics inference picks up the right threat model, norm,
and families. Override the algorithm at runtime via the YAML or CLI:

```bash
uv run raitap +robustness=torchattacks_pgd robustness.pgd.algorithm=DeepFool
uv run raitap +robustness=foolbox_lin_pgd robustness.linf_pgd.algorithm=L2PGD
```

Add an integration test that asserts the verdicts and per-sample distance look
sane on a tiny CNN. Reference
`src/raitap/robustness/assessors/tests/test_torchattacks_assessor.py` for the
pattern.

## Adding a new framework

To integrate a new robustness framework:

1. **Implement the adapter.** Pick `EmpiricalAttackAssessor` (subclasses
   implement `generate_adversarial`) or `FormalVerificationAssessor`
   (subclasses implement `verify_sample`).
2. **Declare `algorithm_registry`** as a `ClassVar[Mapping[str,
   AssessorSemanticsHints]]` on the new adapter — semantics inference reads
   this without any framework branching in `semantics.py`.
3. **Set `budget_kwarg_source`** when the library reads budget kwargs at call
   time (`"call_kwargs"`); it defaults to `"init_kwargs"`.
4. **Override `check_backend_compat`** if the library imposes backend
   constraints (e.g. white-box attacks require `supports_torch_autograd`).
5. **Add an extras group** in `pyproject.toml` and a YAML preset under
   `src/raitap/configs/robustness/`.
6. **Append the library's import name to `THIRD_PARTY_LIBS`** in
   `src/raitap/robustness/__init__.py`. `raitap.utils.diagnostics` consumes
   that set to attribute warnings/errors emitted from inside the library to
   a "via &lt;lib&gt;" chip and the frameworks-and-libraries docs page.
7. **Add adapter tests** under `assessors/tests/`.

The framework-owned base classes do the rest. No core change to
`RobustnessResult`, the factory, the pipeline, or reporting is required.

## Adding a new visualiser

Subclass `BaseRobustnessVisualiser`. Set `supported_method_kinds` so the
factory rejects mismatched assessor pairings at parse time. For image
visualisers, call `_require_image_modality(result, type(self).__name__)` in
`visualise()` so the (B, C, H, W) layout assumption is enforced. Reuse
`_signed_perturbation_heatmap` from
`visualisers/empirical/image_pair_visualiser.py` when rendering signed
perturbation deltas — it preserves the sign of the dominant channel without
collapsing opposing-sign channels to ~0.
