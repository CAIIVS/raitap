---
name: ONNX pipeline abstraction
overview: Linear implementation plan for native ONNX support using `ModelBackend`, `TorchBackend`, and `OnnxBackend`, with explainer-owned ONNX compatibility lists, explicit factory validation order, and removal of the old converter-based design.
todos:
  - id: phase-1-allowlist-exception
    content: Add `algorithm_allowlist.py`, add `ExplainerBackendIncompatibilityError`, and route visualiser compat through the shared allow-list helper.
    status: completed
  - id: phase-2-backend-abstraction
    content: Introduce `ModelBackend` and `TorchBackend`, change `Model` to store `backend`, and move pipeline forward calls to `model.backend(...)`.
    status: completed
  - id: phase-3-explainer-factory
    content: Add `check_backend_compat()` to explainers, add ONNX allow-lists to Captum/Shap explainers, and make the factory validate visualisers then backend before `explain()`.
    status: completed
  - id: phase-4-onnx-backend
    content: Add `onnxruntime` support, implement `OnnxBackend`, and load `.onnx` models natively.
    status: completed
  - id: phase-5-onnx-explainers
    content: Implement ONNX explanation bridge via `as_model_for_explanation()`, Captum ORT wrapper, and SHAP KernelExplainer path.
    status: completed
  - id: phase-6-tracking
    content: Branch tracking/logging by backend and use `mlflow.onnx.log_model` for ONNX.
    status: completed
  - id: phase-7-remove-legacy
    content: Remove `FormatConverter` / `CONVERTERS` design, update docs, and rewrite tests for the backend-based API.
    status: completed
isProject: false
---

# ONNX Support Without Conversion

## Final Decisions

- Do **not** convert ONNX to `.pth`.
- Use **Option B**: `ModelBackend` + `TorchBackend` + `OnnxBackend`.
- Store ONNX compatibility lists **inside each explainer class**, not in backends and not in a central compatibility registry.
- Each explainer implements `check_backend_compat(self, backend)`.
- Use one small agnostic helper for the repeated rule:
  - empty `frozenset` means "no restriction"
  - non-empty `frozenset` means `algorithm` must be a member
- Preserve current factory behavior that resolves `call:` data-source dicts through `load_tensor_from_source(...)` (e.g. `background_data: { source: ..., n_samples: ... }`).
- Preserve current `BaseExplainer` batching/progress behavior:
  - `batch_size`
  - `max_batch_size`
  - `show_progress`
  - `progress_desc`
  - `background_data` must remain exempt from per-batch slicing
- Preserve current pipeline behavior that passes `sample_names=getattr(data, "sample_ids", None)` into the explanation flow.
- Factory validation order is fixed:
  1. create explainer
  2. create visualisers
  3. validate visualiser compatibility
  4. validate backend compatibility
  5. run `explain()`
- `ModelBackend` must expose:
  - `supports_torch_autograd: bool`
  - `as_model_for_explanation() -> nn.Module`
- Logging contract is fixed:
  - Torch backend -> `mlflow.pytorch.log_model`
  - ONNX backend -> `mlflow.onnx.log_model`

## Backend Contracts

### `ModelBackend`

Add a new backend abstraction in a new file such as [`src/raitap/models/backend.py`](src/raitap/models/backend.py).

Required API:

```python
class ModelBackend(ABC):
    supports_torch_autograd: bool

    @abstractmethod
    def __call__(self, inputs: torch.Tensor) -> Any:
        ...

    @abstractmethod
    def as_model_for_explanation(self) -> torch.nn.Module:
        ...
```

Semantics:

- `__call__` is the backend-agnostic inference entrypoint used by the pipeline.
- `as_model_for_explanation()` returns the object passed into `explainer.explain(...)`.
- For Torch this is the real `nn.Module`.
- For ONNX this is a thin `nn.Module` wrapper around ORT.

### `TorchBackend`

Implement in [`src/raitap/models/backend.py`](src/raitap/models/backend.py).

Required behavior:

- stores the real `nn.Module`
- `supports_torch_autograd = True`
- `__call__` delegates to module forward
- `as_model_for_explanation()` returns the stored `nn.Module`

### `OnnxBackend`

Implement in [`src/raitap/models/backend.py`](src/raitap/models/backend.py).

Required behavior:

- stores `onnxruntime.InferenceSession`
- stores resolved input / output names needed for inference
- `supports_torch_autograd = False`
- `__call__`:
  1. accepts `torch.Tensor`
  2. moves / detaches to CPU as needed
  3. converts to numpy
  4. runs `session.run(...)`
  5. converts the primary output back to `torch.Tensor`
- `as_model_for_explanation()` returns a thin `nn.Module` wrapper whose `forward()` performs the same ORT call

Do **not** merge `OnnxBackend` without `as_model_for_explanation()` being implemented. `.onnx` forward and `.onnx` explanation must land together.

## Explainer Compatibility

### Where compatibility lives

Compatibility data must live **on the explainer classes**:

- [`src/raitap/transparency/explainers/captum_explainer.py`](src/raitap/transparency/explainers/captum_explainer.py)
- [`src/raitap/transparency/explainers/shap_explainer.py`](src/raitap/transparency/explainers/shap_explainer.py)

Add class attributes:

```python
class CaptumExplainer(BaseExplainer):
    ONNX_COMPATIBLE_ALGORITHMS: frozenset[str] = ...

class ShapExplainer(BaseExplainer):
    ONNX_COMPATIBLE_ALGORITHMS: frozenset[str] = ...
```

### Shared allow-list utility

Create a tiny helper module:

- [`src/raitap/transparency/algorithm_allowlist.py`](src/raitap/transparency/algorithm_allowlist.py)

Required function:

```python
def ensure_algorithm_in_allowlist(
    algorithm: str,
    allowlist: frozenset[str],
    *,
    error_cls: type[Exception],
    **error_kwargs: Any,
) -> None:
    ...
```

Required behavior:

- if `allowlist` is empty: return
- if `algorithm in allowlist`: return
- otherwise: raise `error_cls(**error_kwargs)`

This exact helper is used by:

- explainer backend compatibility checks
- visualiser compatibility checks if the existing logic matches exactly

### New explainer method

Add to [`src/raitap/transparency/explainers/base_explainer.py`](src/raitap/transparency/explainers/base_explainer.py):

```python
def check_backend_compat(self, backend: ModelBackend) -> None:
    return None
```

Subclasses override it.

#### `CaptumExplainer.check_backend_compat`

Implementation in [`src/raitap/transparency/explainers/captum_explainer.py`](src/raitap/transparency/explainers/captum_explainer.py):

1. if `backend.supports_torch_autograd` is `True`, return immediately
2. otherwise call `ensure_algorithm_in_allowlist(...)` with:
   - `algorithm=self.algorithm`
   - `allowlist=type(self).ONNX_COMPATIBLE_ALGORITHMS`
   - `error_cls=ExplainerBackendIncompatibilityError`

#### `ShapExplainer.check_backend_compat`

Implementation in [`src/raitap/transparency/explainers/shap_explainer.py`](src/raitap/transparency/explainers/shap_explainer.py):

1. if `backend.supports_torch_autograd` is `True`, return immediately
2. otherwise call `ensure_algorithm_in_allowlist(...)` with:
   - `algorithm=self.algorithm`
   - `allowlist=type(self).ONNX_COMPATIBLE_ALGORITHMS`
   - `error_cls=ExplainerBackendIncompatibilityError`

### Compatibility matrix

#### SHAP

For `OnnxBackend`, the only allowed SHAP algorithm is:

```python
ONNX_COMPATIBLE_ALGORITHMS = frozenset({"KernelExplainer"})
```

Everything else must fail before explanation.

#### Captum

For `OnnxBackend`, the allowed Captum algorithms are exactly:

```python
ONNX_COMPATIBLE_ALGORITHMS = frozenset({
    "FeatureAblation",
    "FeaturePermutation",
    "Occlusion",
    "ShapleyValueSampling",
    "ShapleyValues",
    "KernelShap",
    "Lime",
})
```

Everything else must fail before explanation.

Do **not** broaden this list during implementation unless the plan is explicitly revised.

## Factory Flow

File:

- [`src/raitap/transparency/factory.py`](src/raitap/transparency/factory.py)

Required final order inside `Explanation.__new__`:

1. read transparency config
2. resolve `algorithm`
3. call `create_explainer(...)`
4. call `create_visualisers(...)`
5. call `check_explainer_visualiser_compat(...)`
6. call `explainer.check_backend_compat(model.backend)`
7. merge `call:` kwargs
8. call:

```python
explainer.explain(
    model.backend.as_model_for_explanation(),
    inputs,
    ...
)
```

The factory must **not** contain backend-specific algorithm lists. It only invokes explainer-owned checks.

The factory **must continue** to resolve `call:` data-source references before calling `explain()`. When the refactor moves code around, preserve the current semantics of `_resolve_call_data_sources(...)` from [`src/raitap/transparency/factory.py`](src/raitap/transparency/factory.py).

## Linear Implementation Steps

## Phase 1 — Shared allow-list helper and exception

### Step 1.1

Create [`src/raitap/transparency/algorithm_allowlist.py`](src/raitap/transparency/algorithm_allowlist.py) with `ensure_algorithm_in_allowlist(...)`.

### Step 1.2

Add `ExplainerBackendIncompatibilityError` and move `VisualiserIncompatibilityError` into:

- [`src/raitap/transparency/exceptions.py`](src/raitap/transparency/exceptions.py)

Implementation rule:

- create `transparency/exceptions.py` as the single home for transparency compatibility / orchestration exceptions
- move `VisualiserIncompatibilityError` out of `visualisers/base_visualiser.py`
- define `ExplainerBackendIncompatibilityError` in the same file
- export both exceptions from [`src/raitap/transparency/__init__.py`](src/raitap/transparency/__init__.py)
- update [`src/raitap/transparency/visualisers/__init__.py`](src/raitap/transparency/visualisers/__init__.py) so it re-exports `VisualiserIncompatibilityError` from the new module, not from `base_visualiser.py`
- update imports so `factory.py`, visualisers, and tests import from `transparency.exceptions` or package exports, not from `visualisers/base_visualiser.py`

### Step 1.3

Refactor `check_explainer_visualiser_compat(...)` in [`src/raitap/transparency/factory.py`](src/raitap/transparency/factory.py) to use the shared helper if the logic matches exactly.

Do **not** remove or regress:

- `compatible_algorithms` semantics on `BaseVisualiser`
- error message shape used by current tests
- package export of `VisualiserIncompatibilityError` from [`src/raitap/transparency/__init__.py`](src/raitap/transparency/__init__.py) after the move to `transparency/exceptions.py`
- re-export of `VisualiserIncompatibilityError` from [`src/raitap/transparency/visualisers/__init__.py`](src/raitap/transparency/visualisers/__init__.py)

### Step 1.4

Run:

- `pytest src/raitap/transparency/tests/test_factory.py`

Do not continue until it passes.

## Phase 2 — Backend abstraction and Torch path

### Step 2.1

Create `ModelBackend` and `TorchBackend` in [`src/raitap/models/backend.py`](src/raitap/models/backend.py).

### Step 2.2

Update [`src/raitap/models/model.py`](src/raitap/models/model.py):

- replace `self.network` with `self.backend`
- `_load_model(...)` must return `ModelBackend`
- `.pth` / `.pt` paths return `TorchBackend(module)`
- torchvision pretrained names return `TorchBackend(module)`

### Step 2.3

Update [`src/raitap/run/pipeline.py`](src/raitap/run/pipeline.py):

- replace `model.network(data_tensor)` with `model.backend(data_tensor)`
- preserve existing `torch.no_grad()` placement
- preserve metrics flow using `extract_primary_tensor(...)`, `metrics_prediction_pair(...)`, and `resolve_metric_targets(...)`
- preserve `sample_names=getattr(data, "sample_ids", None)` when constructing `Explanation(...)`

### Step 2.4

Grep and replace all remaining `model.network` / `self.network` references in `src/` and tests.

### Step 2.5

Run the test suite and fix all Torch-path breakage before adding ONNX:

- `pytest`

## Phase 3 — Explainer-owned backend checks

### Step 3.1

Add `check_backend_compat(...)` to [`src/raitap/transparency/explainers/base_explainer.py`](src/raitap/transparency/explainers/base_explainer.py).

While editing `BaseExplainer`, preserve these existing behaviors:

- `_compute_with_optional_batches(...)`
- `_pop_batch_size(...)`
- `_pop_progress_settings(...)`
- `_slice_kwargs_for_batch(...)`
- `_VISUALISATION_ONLY_KWARGS` filtering

If backend context is needed inside explainers, extend the API in a way that does **not** break those behaviors. Example acceptable approach:

- `explain(..., backend=model.backend, **kwargs)`
- thread `backend` through to `compute_attributions(...)` only where needed

Unacceptable approach:

- rewriting `BaseExplainer.explain(...)` in a way that drops batching / progress support already covered by tests and docs

### Step 3.2

Add `ONNX_COMPATIBLE_ALGORITHMS` to:

- [`src/raitap/transparency/explainers/captum_explainer.py`](src/raitap/transparency/explainers/captum_explainer.py)
- [`src/raitap/transparency/explainers/shap_explainer.py`](src/raitap/transparency/explainers/shap_explainer.py)

using the exact literals from this plan.

### Step 3.3

Implement `check_backend_compat(...)` overrides in both explainers.

### Step 3.4

Update [`src/raitap/transparency/factory.py`](src/raitap/transparency/factory.py):

- after visualiser compatibility, call `explainer.check_backend_compat(model.backend)`

### Step 3.5

Change the model passed to explainers from `model.network` to:

```python
model.backend.as_model_for_explanation()
```

Preserve the current `ExplanationResult` metadata path in `BaseExplainer.explain(...)`, including passthrough of visualization-only kwargs such as `sample_names`.

### Step 3.6

Add tests proving:

- Torch backend passes `check_backend_compat`
- ONNX-style backend blocks `IntegratedGradients`
- ONNX-style backend blocks `GradientExplainer`
- ONNX-style backend allows `FeatureAblation`
- ONNX-style backend allows `KernelExplainer`

Current codebase note:

- [`src/raitap/transparency/tests/test_factory.py`](src/raitap/transparency/tests/test_factory.py) currently builds `SimpleNamespace(network=...)` test doubles
- those must be updated to `backend=...` doubles that expose `as_model_for_explanation()`
- the same applies anywhere else tests currently patch or assert against `model.network`

## Phase 4 — Native ONNX loading

### Step 4.1

Add a new optional dependency group in [`pyproject.toml`](pyproject.toml):

- `onnx`

Include at least:

- `onnxruntime`
- `onnx`

Rationale:

- `onnxruntime` is required for inference
- `onnx` is expected for ONNX model handling in tests / fixtures and may also be needed by `mlflow.onnx.log_model`

Do not guess versions; add current package-manager defaults unless project constraints force otherwise.

### Step 4.2

Implement `OnnxBackend` in [`src/raitap/models/backend.py`](src/raitap/models/backend.py).

### Step 4.3

Teach [`src/raitap/models/model.py`](src/raitap/models/model.py) to load `.onnx` into `OnnxBackend`.

Import `onnxruntime` lazily and raise a clear install message if unavailable.

When updating `Model`, also update current stale user-facing messages and docstrings that still say:

- foreign formats are converted to `.pth`
- supported formats come from `CONVERTERS`

### Step 4.4

Verify plain forward-only pipeline behavior with an ONNX fixture before explanation work.

## Phase 5 — ONNX explanation bridge

### Step 5.1

Implement the private ORT-backed `nn.Module` wrapper used by `OnnxBackend.as_model_for_explanation()`.

### Step 5.2

Wire `OnnxBackend.as_model_for_explanation()` to return that wrapper.

### Step 5.3

Confirm the factory uses `as_model_for_explanation()` everywhere and does not branch on backend type.

### Step 5.4

Update [`src/raitap/transparency/explainers/shap_explainer.py`](src/raitap/transparency/explainers/shap_explainer.py):

- when backend is ONNX / non-autograd, only run `KernelExplainer`
- use an ORT-backed callable returning numpy-compatible outputs
- keep the existing tensor conversion / target post-processing logic consistent
- preserve current background-data semantics:
  - if `background_data` is absent for algorithms that require it, keep the current warning/fallback behavior where applicable

### Step 5.5

Update [`src/raitap/transparency/explainers/captum_explainer.py`](src/raitap/transparency/explainers/captum_explainer.py):

- keep dynamic `getattr(captum.attr, algorithm)`
- rely on the pre-check to block disallowed ONNX algorithms
- verify at least one allowed ONNX Captum method actually runs through the wrapper
- do not regress the existing constructor-vs-call split already implemented in `factory.py`

### Step 5.6

Add or generate a tiny ONNX fixture for tests.

Required coverage:

- ONNX + Captum `FeatureAblation`
- ONNX + SHAP `KernelExplainer`
- blocked ONNX + Captum `IntegratedGradients`
- blocked ONNX + SHAP `GradientExplainer`

Do not proceed to tracking until these tests pass.

## Phase 6 — Tracking

### Step 6.1

Update [`src/raitap/tracking/base_tracker.py`](src/raitap/tracking/base_tracker.py) so `log_model(...)` accepts the new backend-based flow.

### Step 6.2

Update [`src/raitap/models/model.py`](src/raitap/models/model.py) and/or [`src/raitap/tracking/mlflow_tracker.py`](src/raitap/tracking/mlflow_tracker.py):

- Torch backend -> `mlflow.pytorch.log_model`
- ONNX backend -> `mlflow.onnx.log_model`

Implementation note:

- because the current plan already adds the `onnx` optional dependency in Phase 4, do not introduce a second competing dependency path for MLflow ONNX support

### Step 6.3

Rewrite tracker tests accordingly.

## Phase 7 — Remove legacy converter-based design

### Step 7.1

Replace `FormatConverter` / `CONVERTERS` with a loader design that returns `ModelBackend`, not `nn.Module`.

Files likely affected:

- [`src/raitap/models/converters.py`](src/raitap/models/converters.py)
- [`src/raitap/models/model.py`](src/raitap/models/model.py)

### Step 7.2

Update stale docs and messages that still describe conversion to `.pth`.

At minimum:

- [`src/raitap/models/__init__.py`](src/raitap/models/__init__.py)
- [`src/raitap/models/model.py`](src/raitap/models/model.py)
- [`src/raitap/transparency/README.md`](src/raitap/transparency/README.md) if batching/progress or ONNX explainer support docs become inaccurate

### Step 7.3

Rewrite affected tests:

- [`src/raitap/models/tests/test_models.py`](src/raitap/models/tests/test_models.py)
- [`src/raitap/models/tests/test_model_class.py`](src/raitap/models/tests/test_model_class.py)
- transparency tests touched by backend changes
- tracking tests touched by logging changes

### Step 7.4

Delete any dead ONNX -> PyTorch conversion remnants still present on the branch.

### Step 7.5

Run final verification:

- `pytest`
- `ruff`
- `pyright`

Do not consider the work complete until all three pass or their failures are explicitly understood and documented.

## Non-Negotiable Implementation Rules

- Do **not** put ONNX compatibility data in `raitap.models`.
- Do **not** put backend-specific allow-lists in `factory.py`.
- Do **not** broaden the ONNX compatibility matrix during implementation.
- Do **not** merge ONNX forward support without ONNX explanation bridge support.
- Do **not** leave `model.network` as a parallel API after the refactor.
- Do **not** place transparency compatibility exceptions in visualiser files; `VisualiserIncompatibilityError` and `ExplainerBackendIncompatibilityError` must both live in [`src/raitap/transparency/exceptions.py`](src/raitap/transparency/exceptions.py).
- Do **not** forget that [`src/raitap/transparency/visualisers/__init__.py`](src/raitap/transparency/visualisers/__init__.py) currently re-exports `VisualiserIncompatibilityError`; preserve that public surface after moving the class.

## Appendix

### `CaptumExplainer`

```python
ONNX_COMPATIBLE_ALGORITHMS: frozenset[str] = frozenset({
    "FeatureAblation",
    "FeaturePermutation",
    "Occlusion",
    "ShapleyValueSampling",
    "ShapleyValues",
    "KernelShap",
    "Lime",
})
```

### `ShapExplainer`

```python
ONNX_COMPATIBLE_ALGORITHMS: frozenset[str] = frozenset({"KernelExplainer"})
```

## Developer Handoff Checklist

Use this as the execution checklist. Do not skip ahead if the validation step for the current item fails.

### Before touching code

1. Read this full plan once.
2. Confirm the branch still contains no active ONNX -> PyTorch conversion work that should be preserved.
3. Grep these symbols before changing anything:
   - `model.network`
   - `VisualiserIncompatibilityError`
   - `CONVERTERS`
   - `FormatConverter`
   - `check_explainer_visualiser_compat`

### Phase 1 completion gate

1. Add `src/raitap/transparency/algorithm_allowlist.py`.
2. Create `src/raitap/transparency/exceptions.py`.
3. Move `VisualiserIncompatibilityError` there.
4. Add `ExplainerBackendIncompatibilityError` there.
5. Update imports / exports.
6. Refactor `check_explainer_visualiser_compat(...)` to use the shared helper if logic matches exactly.
7. Run:
   - `pytest src/raitap/transparency/tests/test_factory.py`

Only continue if that test target passes.

### Phase 2 completion gate

1. Add `ModelBackend` and `TorchBackend`.
2. Replace `Model.network` with `Model.backend`.
3. Update pipeline forward to `model.backend(data_tensor)`.
4. Replace remaining `model.network` references in source and tests.
5. Run:
   - `pytest`

Only continue if the repo passes on the Torch path before any ONNX backend work.

### Phase 3 completion gate

1. Add `check_backend_compat()` to `BaseExplainer`.
2. Add `ONNX_COMPATIBLE_ALGORITHMS` to `CaptumExplainer` and `ShapExplainer`.
3. Implement explainer-owned backend checks.
4. Make `factory.py` call:
   - `check_explainer_visualiser_compat(...)`
   - `explainer.check_backend_compat(model.backend)`
   - `explainer.explain(model.backend.as_model_for_explanation(), ...)`
5. Add unit tests for allowed / blocked backend-algorithm pairs.
6. Run:
   - `pytest src/raitap/transparency/tests/test_factory.py`
   - `pytest src/raitap/transparency/tests/test_e2e_integration.py`

Only continue if Torch still works and compat failures happen before attribution execution.

### Phase 4 + 5 completion gate

These phases must land together.

1. Add `onnxruntime` optional dependency group.
2. Implement `OnnxBackend`.
3. Implement the ORT-backed wrapper returned by `as_model_for_explanation()`.
4. Load `.onnx` into `OnnxBackend`.
5. Implement SHAP ONNX path for `KernelExplainer`.
6. Implement Captum ONNX path through the wrapper for the allow-listed algorithms.
7. Add tiny ONNX test fixture(s).
8. Add tests covering:
   - ONNX + Captum `FeatureAblation`
   - ONNX + SHAP `KernelExplainer`
   - blocked ONNX + Captum `IntegratedGradients`
   - blocked ONNX + SHAP `GradientExplainer`
9. Run:
   - `pytest`

Only continue if native `.onnx` forward and explanation both work in the same branch state.

### Phase 6 completion gate

1. Update tracker interfaces for backend-based logging.
2. Branch logging by backend:
   - Torch -> `mlflow.pytorch.log_model`
   - ONNX -> `mlflow.onnx.log_model`
3. Rewrite tracker tests.
4. Run:
   - `pytest src/raitap/tracking/tests`

Only continue if tracking passes for both backend families.

### Phase 7 completion gate

1. Remove `FormatConverter` / `CONVERTERS` design.
2. Replace it with backend-returning loading.
3. Update all stale `.pth conversion` docs/messages.
4. Rewrite affected model / transparency / tracking tests.
5. Delete ONNX conversion remnants.
6. Run:
   - `pytest`
   - `ruff`
   - `pyright`

The refactor is only done when all three succeed, or any remaining failures are clearly pre-existing and documented.

### Final review checklist

Before opening a PR, verify all of the following are true:

1. `model.network` no longer exists as an active API.
2. `VisualiserIncompatibilityError` and `ExplainerBackendIncompatibilityError` both live in `src/raitap/transparency/exceptions.py`.
3. No ONNX compatibility lists exist outside `CaptumExplainer` / `ShapExplainer`.
4. `factory.py` does not own backend-specific allow-lists.
5. `.onnx` does not rely on ONNX -> PyTorch conversion.
6. `as_model_for_explanation()` is the single bridge from backend to explainer input.
7. Torch behavior remains intact for existing transparency configs.
