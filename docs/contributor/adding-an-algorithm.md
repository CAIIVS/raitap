---
title: "Adding an algorithm"
description: "How to expose a new algorithm of an already-wrapped library — extend the existing adapter's algorithm_registry instead of adding a new adapter class."
myst:
  html_meta:
    "description": "How to expose a new algorithm of an already-wrapped library — extend the existing adapter's algorithm_registry instead of adding a new adapter class."
---

# Adding an algorithm

If the library is already wrapped (e.g. Captum, SHAP, torchattacks, foolbox), exposing a new algorithm is a one-decorator-kwarg edit on the existing adapter — no new class, no new file, no `pyproject.toml` change. If the library is **not** wrapped yet, see [Adding an adapter](adding-an-adapter.md) instead.

The walkthrough below uses Captum and a hypothetical new method `NewMethod`.

## 1. Find the adapter

Adapter classes live under `src/raitap/<module>/<subdir>/`. Look for the file matching the wrapped library (e.g. `captum_explainer.py`, `torchattacks_assessor.py`). The class is decorated with `@adapters.<family>(...)`.

## 2. Add the algorithm to `algorithm_registry`

`algorithm_registry` is a decorator kwarg on the adapter — a mapping of algorithm name → semantics. Add one entry.

**Transparency explainer** (`captum_explainer.py`):

```python
from raitap import adapters

@adapters.transparency(
    registry_name="captum",
    library="captum",
    algorithm_registry={
        "IntegratedGradients": frozenset({MethodFamily.GRADIENT}),
        # ... existing entries ...
        "NewMethod": frozenset({MethodFamily.GRADIENT, MethodFamily.PERTURBATION}),
    },
    onnx_compatible_algorithms=frozenset({...}),
)
class CaptumExplainer(AttributionOnlyExplainer): ...
```

**Robustness assessor** (`torchattacks_assessor.py`):

```python
from raitap import adapters

@adapters.robustness(
    registry_name="torchattacks",
    library="torchattacks",
    algorithm_registry={
        # ... existing entries ...
        "NewAttack": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,
            families=frozenset({"gradient_sign"}),
        ),
    },
)
class TorchattacksAssessor(EmpiricalAttackAssessor): ...
```

The map value carries the semantics RAITAP tracks and reports on:
- **Transparency** → `frozenset[MethodFamily]`. New `MethodFamily` values go in `src/raitap/transparency/contracts.py`.
- **Robustness** → `AssessorSemanticsHints` (method kind, threat model, objective, norm, family tags). Defined in `src/raitap/robustness/semantics.py`.

A missing entry means the algorithm cannot be selected via config — the family's runtime check fails fast.

## 3. ONNX compatibility (transparency only, optional)

If the new algorithm runs on ONNX-exported models (not just torch), add it to `onnx_compatible_algorithms`:

```python
@adapters.transparency(
    ...,
    onnx_compatible_algorithms=frozenset({"Occlusion", "FeatureAblation", "NewMethod"}),
)
```

Pass `from raitap.transparency import ALL` (or `raitap.robustness.ALL`) instead if **every** algorithm in `algorithm_registry` is ONNX-compatible. Omit the kwarg entirely if no algorithms are ONNX-compatible (the default).

The default `check_backend_compat` enforces this allowlist — algorithms not in the set raise `ExplainerBackendIncompatibilityError` (or `AssessorBackendIncompatibilityError`) when the user picks an ONNX backend.

## 4. Tests

Add a unit test next to the adapter (`src/raitap/<module>/<subdir>/tests/test_<adapter>.py`) that:

1. Constructs the adapter with `algorithm="NewMethod"` and minimal kwargs.
2. Runs the happy path (`compute_attributions(...)` / `generate_adversarial(...)`).
3. Asserts the output shape/type matches the contract.

If the algorithm has unusual kwargs (e.g. a custom `baselines=` shape), add an edge-case test for those too.

Reuse shared helpers instead of re-rolling fixtures: `from raitap.testing import make_tiny_classifier, make_app_config, requires` and the root `seeded` fixture.

If the wrapped library is **deterministic** (Captum, torchattacks with `random_start=False`, foolbox, Marabou — not sampling-based SHAP), add a **parity test** marked `@pytest.mark.e2e @pytest.mark.parity` that asserts `torch.allclose(raitap_output, direct_library_call)` for the same config. Use at least one non-default kwarg so a silently-dropped kwarg fails the assertion. This proves raitap relays the library faithfully.

The family E2E matrix parametrises over algorithm names — add an entry to keep coverage complete:

- **Transparency**: `src/raitap/transparency/tests/e2e_case_matrix.py::MATRIX_CASES`. Add a `MatrixCase(id="...", framework=..., algorithm="NewMethod", ...)`.
- **Robustness**: `src/raitap/robustness/tests/e2e_assessor_matrix.py::MATRIX_CASES`. Add an `AssessorMatrixCase(id="...", family=..., algorithm="NewAlgo", needs_extra=..., constructor_kwargs={...})`. Keep `constructor_kwargs` minimal (low `steps`, low `n_queries`) — the matrix is a wire-up smoke test, not a behaviour-sensitivity test. Each case must finish in under ~5s on CI.

## 5. Docs

Add a row to `docs/modules/<module>/frameworks-and-libraries.md` for the new algorithm so it surfaces in the user-facing "does raitap support X?" lookup. Mention the families it belongs to and whether it is ONNX-compatible.

That is the whole change. No `pyproject.toml`, no decorator changes, no factory edits.
