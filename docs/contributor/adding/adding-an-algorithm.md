---
title: "Adding an algorithm"
description: "How to expose a new algorithm of an already-wrapped library: extend the existing adapter's algorithm_registry instead of adding a new adapter class."
myst:
  html_meta:
    "description": "How to expose a new algorithm of an already-wrapped library: extend the existing adapter's algorithm_registry instead of adding a new adapter class."
---

# Adding an algorithm

If the library is already wrapped (e.g. Captum, SHAP, torchattacks, foolbox), exposing a new algorithm is a one-decorator-kwarg edit on the existing adapter: no new class, no new file, no `pyproject.toml` change. If the library is **not** wrapped yet, see [Adding an adapter](adding-an-adapter.md) instead.

The walkthrough below uses Captum and a hypothetical new method `NewMethod`.

## 1. Find the adapter

Adapter classes live under `src/raitap/<module>/<subdir>/`. Look for the file matching the wrapped library (e.g. `captum_explainer.py`, `torchattacks_assessor.py`). The class is decorated with `@adapters.<family>(...)`.

## 2. Add the algorithm to `algorithm_registry`

`algorithm_registry` is a decorator kwarg on the adapter: a mapping of algorithm name to semantics. Add one entry.

**Transparency explainer** (`captum_explainer.py`):

```python
from raitap import adapters

@adapters.transparency(
    registry_name="captum",
    import_name="captum",
    algorithm_registry={
        "IntegratedGradients": ExplainerAlgorithmSpec(
            {MethodFamily.GRADIENT},
            baseline_default=BaselineMode.ZERO,
            requires={Capability.AUTOGRAD},  # gradient method
        ),
        # ... existing entries ...
        "NewMethod": ExplainerAlgorithmSpec(
            {MethodFamily.GRADIENT, MethodFamily.PERTURBATION},
            requires={Capability.AUTOGRAD},  # gradient method
        ),
    },
    baseline_kwarg_name="baselines",
)
class CaptumExplainer(AttributionOnlyExplainer): ...
```

**Robustness assessor** (`torchattacks_assessor.py`):

```python
from raitap import adapters

@adapters.robustness(
    registry_name="torchattacks",
    import_name="torchattacks",
    algorithm_registry={
        # ... existing entries ...
        "NewAttack": AssessorAlgorithmSpec(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,
            families={"gradient_sign"},
            requires={Capability.AUTOGRAD},  # gradient-based attack
        ),
    },
)
class TorchattacksAssessor(EmpiricalAttackAssessor): ...
```

The map value carries the semantics RAITAP tracks and reports on:
- **Transparency** → `ExplainerAlgorithmSpec` (`families: AbstractSet[MethodFamily]` + optional `baseline_default` + optional `requires`). New `MethodFamily` values go in `src/raitap/transparency/contracts.py`.
- **Robustness** → `AssessorAlgorithmSpec` (assessment kind, threat model, objective, norm, family tags, optional `requires`). Defined in `src/raitap/robustness/semantics.py`.

A missing entry means the algorithm cannot be selected via config.

## 3. Backend capability (`requires`)

The `requires` field on `ExplainerAlgorithmSpec` / `AssessorAlgorithmSpec` declares what the algorithm needs from the backend. The rule: an algorithm runs on a backend iff `algorithm.requires <= backend.provides`. The gate is enforced automatically by inherited `AdapterMixin.check_backend_compat`: you write nothing extra.

| Algorithm type | `requires` value | Effect |
|---|---|---|
| Gradient-based (IntegratedGradients, PGD, FGSM, ...) | `{Capability.AUTOGRAD}` | Blocked on ONNX (forward-only) backends |
| Model-agnostic (SHAP KernelExplainer, Occlusion, FeatureAblation, ...) | `frozenset()` (default) | Runs on any backend, including ONNX |

Import: `from raitap.types import Capability`. `Capability.AUTOGRAD` and `Capability.TREE_MODEL` are live gate values; `PREDICT_PROBA` is provided by tree backends but is read by the forward pass, not used as an algorithm gate. See {doc}`../capabilities` for the full capability reference.

When `requires - backend.provides` is non-empty, `BackendIncompatibilityError` is raised (`from raitap.utils.errors import BackendIncompatibilityError`; also re-exported from `raitap.robustness` and `raitap.transparency`).

## 4. Baseline default (transparency only, optional)

Attribution methods that take a reference input (Integrated Gradients via `baselines=` and SHAP via `background_data=`) have that baseline recorded in `metadata.json` and the report (issue #210), and users set it library-agnostically via `raitap.baseline`. Three declarations drive this:

- `baseline_kwarg_name`: a `@adapters.transparency` decorator kwarg naming the call kwarg that holds the reference (`"baselines"` for Captum, `"background_data"` for SHAP). Omitted (the default) means the family takes no baseline. It is per-**adapter** (one library, one kwarg name), and is where `raitap.baseline` gets routed.
- `ExplainerAlgorithmSpec.baseline_default`: the per-**algorithm** implicit default mode, used when the user omits the kwarg. Lives on the algorithm's registry entry because one adapter wraps many algorithms, most of which take no baseline (so they leave it `None`).
- `ExplainerAlgorithmSpec.baseline_cardinality`: `BaselineCardinality.SINGLE` (one broadcast reference, e.g. IG) or `SET` (a sample distribution, e.g. SHAP). Used only to *warn* on a mismatched `raitap.baseline` (never to reshape it); leave `None` to skip the check.

If your new algorithm takes a baseline **and** has a meaningful default when the user omits it, set `baseline_default` (and, ideally, `baseline_cardinality`) on its registry entry:

```python
@adapters.transparency(
    registry_name="captum",
    baseline_kwarg_name="baselines",
    algorithm_registry={
        "IntegratedGradients": ExplainerAlgorithmSpec(
            {MethodFamily.GRADIENT},
            baseline_default=BaselineMode.ZERO,
            baseline_cardinality=BaselineCardinality.SINGLE,
            requires={Capability.AUTOGRAD},  # gradient method
        ),
        "NewMethod": ExplainerAlgorithmSpec(
            {MethodFamily.GRADIENT},
            baseline_default=BaselineMode.ZERO,
            baseline_cardinality=BaselineCardinality.SINGLE,
            requires={Capability.AUTOGRAD},  # gradient method
        ),
    },
)
class CaptumExplainer(AttributionOnlyExplainer): ...
```

Nothing to do if your algorithm only uses a baseline when the user supplies one (the kwarg-present path records it as `configured`/`user_tensor` automatically), or if it takes no reference at all (Saliency, GradCam): leave `baseline_default` unset (`None`).

## 5. Tests

Add a unit test next to the adapter (`src/raitap/<module>/<subdir>/tests/test_<adapter>.py`) that:

1. Constructs the adapter with `algorithm="NewMethod"` and minimal kwargs.
2. Runs the happy path (`compute_attributions(...)` / `_default_invoke(ctx)` via `generate_adversarial(...)`).
3. Asserts the output shape/type matches the contract.

If the algorithm has unusual kwargs (e.g. a custom `baselines=` shape), add an edge-case test for those too.

Reuse shared helpers instead of re-rolling fixtures: `from raitap.testing import make_tiny_classifier, make_app_config, requires` and the root `seeded` fixture.

If the wrapped library is **deterministic** (Captum, torchattacks with `random_start=False`, foolbox, Marabou; not sampling-based SHAP), add a **parity test** marked `@pytest.mark.e2e @pytest.mark.parity` that asserts `torch.allclose(raitap_output, direct_library_call)` for the same config. Use at least one non-default kwarg so a silently-dropped kwarg fails the assertion. This proves raitap relays the library faithfully.

The family E2E matrix parametrises over algorithm names. Add an entry to keep coverage complete:

- **Transparency**: `src/raitap/transparency/tests/e2e_case_matrix.py::MATRIX_CASES`. Add a `MatrixCase(id="...", framework=..., algorithm="NewMethod", ...)`.
- **Robustness**: `src/raitap/robustness/tests/e2e_assessor_matrix.py::MATRIX_CASES`. Add an `AssessorMatrixCase(id="...", family=..., algorithm="NewAlgo", needs_extra=..., constructor_kwargs={...})`. Keep `constructor_kwargs` minimal (low `steps`, low `n_queries`): the matrix is a wire-up smoke test, not a behaviour-sensitivity test. Each case must finish in under ~5s on CI.

## 5b. Adding a Quantus metric (transparency evaluation only)

The explanation-quality evaluator (`raitap.transparency.QuantusEvaluator`,
issue #341) follows the same registry pattern, but the registry lives in
`src/raitap/transparency/evaluation/evaluators/quantus_evaluator.py`, not on
an explainer. Append one `QuantusMetricSpec` to `_REGISTRY`:

```python
from raitap.transparency.evaluation.contracts import (
    EvalRequirement as Req,
    QuantusCategory as C,
    QuantusMetricSpec as Spec,
)

_REGISTRY: dict[str, Spec] = {
    # ... existing entries ...
    "new_metric": Spec(
        C.FAITHFULNESS,               # one of the 6 QuantusCategory values
        "NewMetric",                  # the Quantus class name, resolved via getattr(quantus, ...)
        frozenset({Req.ATTRIBUTIONS, Req.MODEL}),  # what it needs - see below
        higher_is_better=True,        # True / False / None (no fixed direction)
    ),
}
```

`requires: frozenset[EvalRequirement]` is this registry's equivalent of
`ExplainerAlgorithmSpec.requires` / `AssessorAlgorithmSpec.requires`: it gates
whether the metric runs for a given explanation, checked by
`resolve_metric` in `evaluation/semantics.py`. Pick from:

| `EvalRequirement` | Set when |
| --- | --- |
| `ATTRIBUTIONS` | Always - every metric needs the attribution tensor. Include it on every entry. |
| `MODEL` | The metric calls the model again (most faithfulness, axiomatic metrics). |
| `RE_EXPLAIN` | The metric re-explains under perturbation (robustness, randomisation metrics). Only satisfied when the originating explainer is an `AttributionOnlyExplainer` (Captum, SHAP). |
| `SEGMENTATION` | The metric needs a ground-truth mask (localisation metrics). No mask source exists yet, so any metric requiring this always skips. |
| `BASELINE` | The metric needs the explainer's reference input. |

A metric missing a satisfied requirement is not an error: it is recorded as a
`SkippedMetric` and the run continues. Set `default_kwargs` (a
`Mapping[str, Any]`) for constructor kwargs the metric always needs; users can
still override per-run via `evaluation.constructor.<metric_name>` in config.

No adapter class, no decorator change, no `pyproject.toml` change - the
`quantus` extra and the `@transparency_evaluator` registration on
`QuantusEvaluator` already cover every registry entry. Add a unit test next to
the evaluator (`src/raitap/transparency/tests/evaluation/test_quantus_evaluator.py`)
constructing the evaluator with `metrics=["new_metric"]` against a fake
`quantus` module (see the existing tests for the monkeypatch pattern) and
asserting the returned `EvaluationScore` or `SkippedMetric`.

## 6. Docs

Add a row to `docs/modules/<module>/frameworks-and-libraries.md` for the new algorithm so it surfaces in the user-facing "does raitap support X?" lookup. Mention the families it belongs to and whether it requires autograd (or is model-agnostic).

That is the whole change. No `pyproject.toml`, no decorator changes, no factory edits.

## 7. Invoker override (advanced, rarely needed)

Most algorithms fit the adapter's uniform construct-and-call path. For the rare algorithm with a non-standard lifecycle, `AssessorAlgorithmSpec` / `ExplainerAlgorithmSpec` accepts an `invoker` field. When set, the adapter calls that function instead of its default path: robustness `generate_adversarial` falls back to `_default_invoke` when no `invoker` is set, while transparency `ShapExplainer.compute_attributions` dispatches between its legacy and modern invokers internally.

The generic `Invoker` Protocol lives in `src/raitap/_adapters.py`:

```python
class Invoker(Protocol[CtxT, ResultT]):
    def __call__(self, ctx: CtxT, /) -> ResultT: ...
```

Per-family context dataclasses:

- **Robustness**: `AttackInvokeCtx` in `assessors/base_assessor.py`. Fields:
  `assessor`, `library`, `model`, `inputs`, `targets`, `backend`,
  `call_kwargs`. The `assessor` field gives access to all shared helpers
  (`_rethrow`, `_prepare_inputs_for_forward`, `_maybe_set_targeted`,
  `_extract_scalar_eps`, `_build_criterion`, `_last_success`).
- **Transparency**: `AttributionInvokeCtx` in `explainers/base_explainer.py`.

**When to use it.** The `invoker` field solves one specific problem: an
algorithm whose lifecycle cannot be expressed as construct-then-call. Examples
in the codebase:

- `foolbox.DatasetAttack` needs `.feed(fmodel, inputs)` before running:
  `_dataset_attack_invoker` in `foolbox_assessor.py` handles the two-stage
  lifecycle. The registry entry passes `invoker=_dataset_attack_invoker`.
- SHAP uses two invokers (`_shap_legacy_invoker` / `_shap_modern_invoker`)
  selected per registry entry. This replaced an older `api` flag on the hints.
  Legacy SHAP explainers (KernelExplainer, GradientExplainer, etc.) use the
  legacy path; modern ones (PartitionExplainer, ExactExplainer,
  PermutationExplainer) use the modern path.

**Verification note.** Per-algorithm hints (`norm`, `threat_model`,
`seeding`, `families`) are verified against the installed library source,
not assumed from docs or class names. When adding an invoker, verify the
lifecycle against the installed library's source and add a unit test that
exercises the invoker path directly.
