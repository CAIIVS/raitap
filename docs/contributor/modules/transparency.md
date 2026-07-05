---
title: "Contributing to the transparency module"
description: "Transparency-specific architecture: the three-level explainer hierarchy, the BaseVisualiser semantic contract, and the ExplanationResult typed semantics."
myst:
  html_meta:
    "description": "Transparency-specific architecture: the three-level explainer hierarchy, the BaseVisualiser semantic contract, and the ExplanationResult typed semantics."
---

# Contributing to the transparency module

Generic adapter mechanics are covered elsewhere:

- New library wrapper: {doc}`../adding/adding-an-adapter`.
- New algorithm on an existing wrapper: {doc}`../adding/adding-an-algorithm`.
- New top-level module: {doc}`../adding/adding-a-module`.

This page documents what's *specific* to transparency: the explainer hierarchy, the visualiser semantic contract, and the typed `ExplanationResult.semantics`.

## Explainer hierarchy

Explainers form a three-level hierarchy (see `src/raitap/transparency/explainers/base_explainer.py` and `full_explainer.py`):

```text
BaseExplainer                       # root: owns output_payload_kind + algorithm_registry contract
├── AttributionOnlyExplainer            # you implement compute_attributions(); base owns explain()
│   ├── CaptumExplainer
│   └── ShapExplainer
└── FullExplainer                       # you implement the full explain() pipeline end-to-end
```

- **`BaseExplainer`**: root base class. Owns `output_payload_kind: ClassVar[ExplanationPayloadKind]`
  (default `ATTRIBUTIONS`) and the `algorithm_registry` contract. Backend gating is inherited from
  `AdapterMixin` (`check_backend_compat`), not declared here. Per-algorithm capability requirements
  live on `ExplainerAlgorithmSpec.requires`. Never subclass directly.
- **`AttributionOnlyExplainer`**: extend this when the framework maps cleanly to a single
  `compute_attributions(model, inputs, **kwargs) -> torch.Tensor` call. Batching, normalisation,
  result wrapping, and `write_artifacts` are handled for you. Captum and SHAP both subclass this.
- **`FullExplainer`**: extend this when you own the entire `explain` pipeline yourself
  (data conversion, model invocation, result construction, persistence).

`output_payload_kind: ClassVar[ExplanationPayloadKind]` (default `ATTRIBUTIONS`) records what
artefact shape the explainer emits. It's set via the `@adapters.transparency` decorator kwarg.

### Baseline contract (reference-input methods)

Methods that take a *reference input* (IG `baselines=`, SHAP `background_data=`) document that
baseline in `metadata.json` + the report (issue #210). The user sets it library-agnostically via
`raitap.baseline`, routed to the adapter's own kwarg. Three declarations on the adapter drive this:

| Declaration | Where | Purpose |
| --- | --- | --- |
| `baseline_kwarg_name` | `@adapters.transparency` decorator kwarg | The call kwarg holding the reference (`"baselines"`, `"background_data"`); omitted (default `None`) = no baseline. Per-**adapter**. Also names where `raitap.baseline` is routed. |
| `ExplainerAlgorithmSpec.baseline_default` | per-algorithm `algorithm_registry` entry | Per-**algorithm** implicit default mode (`BaselineMode.ZERO` / `INPUT_BATCH`) used when the kwarg is omitted; `None` when the algorithm takes no baseline. |
| `ExplainerAlgorithmSpec.baseline_cardinality` | per-algorithm `algorithm_registry` entry | `BaselineCardinality.SINGLE` (one broadcast reference, e.g. IG) or `SET` (a sample distribution, e.g. SHAP). Used to *warn* on a mismatched `raitap.baseline` (never to reshape it); `None` skips the check. |
| `ExplainerAlgorithmSpec.seeding` | per-algorithm `algorithm_registry` entry | 3-state RNG-source classification (issue #339), `Seeding = "deterministic" \| "global_rng" \| "self_seeded"`. `deterministic`: no RNG, always bit-reproducible. `global_rng`: draws from the process-global torch/numpy/random RNG (e.g. SHAP `GradientExplainer`, Captum `Lime`); a pinned `seed` config covers it. `self_seeded`: owns a seed parameter that time-defaults (e.g. SHAP `PermutationExplainer`); a pinned `seed` does not reach it. Resolved by `explainer_seeding`, flows onto `ExplanationResult.semantics.seeding`, and drives the reproducibility caveat. Defaults to `"deterministic"`. A derived read-only `stochastic` property (`seeding != "deterministic"`) still exists for callers that only need the binary check. |

Capture happens once at the `AttributionOnlyExplainer.explain` chokepoint via `build_baseline_record`
(`transparency/baselines.py`), which resolves the `BaselineMode` (`configured` / `user_tensor` /
`zero` / `input_batch`), hashes the tensor, and renders an image preview. It is wrapped so a
render/hash failure degrades to no baseline rather than discarding attributions.
See [Adding an algorithm](../adding/adding-an-algorithm.md).

### Structured payloads (extra tuple outputs)

Some library methods return a tuple: the principal attribution plus extra
diagnostics (e.g. Captum's `return_convergence_delta=True` appends a delta). To
capture these as first-class payloads, declare one `StructuredOutputSpec` per
extra positional element on the algorithm's `ExplainerAlgorithmSpec.extra_outputs`:

```python
from raitap.transparency.contracts import (
    ExplainerAlgorithmSpec,
    StructuredOutputSpec,
    StructuredPayloadKind,
)

ExplainerAlgorithmSpec(
    {MethodFamily.GRADIENT},
    extra_outputs=(
        StructuredOutputSpec("convergence_delta", StructuredPayloadKind.CONVERGENCE_DELTA),
    ),
)
```

Tuple position 0 is always the principal attribution. Positions 1.. map
positionally to the declared specs. If the method returns a tuple but the count
of extras does not match `extra_outputs`, the framework raises a `ValueError`
naming the mismatch, so an undeclared extra output never passes silently. The
captured payloads are persisted to `payloads/<name>.pt` and described in
`metadata.json`; see {doc}`../../modules/transparency/output`. `StructuredPayloadKind`
(`CONVERGENCE_DELTA`, `BASE_VALUE`) and `StructuredOutputSpec` live in
`raitap.transparency.contracts`.

To consume payloads in a visualiser, declare
`supported_structured_payload_kinds={StructuredPayloadKind.CONVERGENCE_DELTA}`
(the kinds it needs) via the `@visualisers.transparency` decorator kwarg, then
read `context.structured_payloads` in `visualise()`. `validate_explanation`
rejects the visualiser when the explanation lacks a declared kind. The registered
`StructuredPayloadSummaryVisualiser` (`structured_payload_summary`) is the
reference renderer.

## ShapExplainer internals

`ShapExplainer.compute_attributions` builds an `AttributionInvokeCtx` and dispatches via the
`invoker` field on each registry entry (`ExplainerAlgorithmSpec.invoker`, added in #266). An
unknown algorithm name produces a `None` entry, falling back to `_shap_legacy_invoker` which raises
the helpful "unsupported algorithm" `ValueError`.

- **Legacy path** (`_shap_legacy_invoker` -> `_compute_legacy` -> `.shap_values()` API):
  `GradientExplainer`, `DeepExplainer`, `KernelExplainer`, `TreeExplainer`, `SamplingExplainer`.
  Constructs the SHAP explainer with the background tensor and calls `.shap_values()` directly.
- **Modern path** (`_shap_modern_invoker` -> `_compute_modern` -> `__call__ -> Explanation` API):
  `PartitionExplainer`, `ExactExplainer`, `PermutationExplainer`. Calls `_build_masker` to select a
  per-modality masker, wraps the predict callable via `_modern_predict_fn`, constructs the
  explainer, and calls it with numpy inputs.

### `_build_masker`

Selects the masker based on `input_spec.kind`:

- `IMAGE`: `shap.maskers.Image("inpaint_telea", (h, w, c))`. Requires `opencv-python` (included in
  the `shap` extra) and an NCHW shape in `input_spec`.
- `TABULAR`: `shap.maskers.Partition(background_np)`.

Other modalities raise `ValueError`. `input_spec` is threaded into `compute_attributions` from the
`AttributionOnlyExplainer.explain` chokepoint via `infer_input_spec`.

### `_normalise_modern_explanation`

Maps the raw `Explanation.values` (class-last layout) to an input-shaped float32 tensor:

1. Cast to float32.
2. Select the target class with `_select_target_attributions` (shared with the legacy path).
3. For image inputs with a class-selected 4-D result, permute NHWC to NCHW to match RAITAP's tensor convention.

## Visualiser semantic contract

All visualisers extend `BaseVisualiser` (`src/raitap/transparency/visualisers/base_visualiser.py`).
On top of `visualise(...) -> Figure` and the optional `save(...)`, each visualiser must declare its
semantic compatibility via ClassVars. The runtime validates these against `ExplanationResult.semantics`
before calling `visualise`.

| ClassVar | Type | Purpose |
| --- | --- | --- |
| `supported_payload_kinds` | `frozenset[ExplanationPayloadKind]` | Payload categories the visualiser can render. |
| `supported_scopes` | `frozenset[ExplanationScope]` | Explanation scopes the visualiser can consume (e.g. `LOCAL` for per-sample attributions). |
| `supported_output_spaces` | `frozenset[ExplanationOutputSpace]` | Attribution coordinate spaces the visualiser handles. |
| `supported_method_families` | `frozenset[MethodFamily]` | Method families the visualiser understands. |
| `compatible_algorithms` | `frozenset[str]` | Optional algorithm allowlist; empty = all algorithms. |
| `supported_structured_payload_kinds` | `frozenset[StructuredPayloadKind]` | Structured payload kinds the visualiser can render. Non-empty means `validate_explanation` rejects an explanation that carries none of the declared kinds (any-of, like `supported_method_families`); the visualiser renders whichever declared kinds are present. Empty = consumes none. |
| `produces_scope` | `ExplanationScope \| None` | Set only when the visualiser *changes* the result scope (e.g. summarising local to aggregated). Leave `None` to preserve the input scope. |
| `scope_definition_step` | `ScopeDefinitionStep \| None` | Where the produced scope was defined; set when `produces_scope` is set. |
| `visual_summary` | `VisualSummarySpec \| None` | Metadata for summary visualisations. |
| `embeds_original_input` | `bool` | Whether the normal layout includes an original-input panel alongside the rendered explanation. |

Plus two instance-level hooks:

- `renders_attribution_only_when_original_hidden() -> bool`: whether `include_original_input=False` still yields a meaningful figure.
- `validate_explanation(explanation, attributions, inputs) -> None`: render-time compatibility validation.

**Rules of thumb**

- Per-sample renderers (heatmap, overlay): preserve scope, leave `produces_scope = None`.
- Summary renderers (SHAP bar/beeswarm, tabular bar): consume local attributions and produce
  `AGGREGATED`. Set `produces_scope = ExplanationScope.AGGREGATED` and
  `scope_definition_step = ScopeDefinitionStep.VISUALISER_SUMMARY`.
- Don't promote arbitrary debug batches or representative montages to `GLOBAL`.
- Image visualisers with `embeds_original_input = True` **must** accept the runtime kwarg
  `include_original_input`. Reporting uses this to render one shared sample thumbnail and suppress
  repeated originals in sample-major compact local report sections. Keep YAML constructor names
  backward-compatible (the built-in image visualisers still accept `include_original_image`).

For the decorator/registration scaffolding (`@visualisers.transparency`, `registry_name`, exports),
see {doc}`../adding/adding-an-adapter`. The bullets above are the transparency-specific additions on top of
that scaffolding.

### Adding an image renderer

`visualise()` forwards style kwargs to the resolved renderer's `draw(**style)`. Any renderer
registered via `@image_renderer` **must** accept `**style` (the `ImageAttributionRenderer`
protocol declares it); a renderer that omits it raises `TypeError` once a user sets `method`.
To participate in the unhonoured-field warning, declare the optional `honours_method` (bool)
and `honoured_signs` (`frozenset[str]`) class constants. They are read via `getattr` with
honour-all defaults, so they are not required.

## Typed semantics contract

`ExplanationResult.semantics` describes the computed explanation artefact. It is a typed contract,
not a narrative description. The contract records the artefact scope, scope definition step, payload
kind, method families, target, sample selection, input metadata, and output-space metadata.

### `ExplanationScope`

Describes the semantic breadth of an explanation or rendered visualisation:

| Scope | Meaning |
| --- | --- |
| `LOCAL` | Explains individual input samples. Current Captum and SHAP attribution explainers produce local explanation artefacts. |
| `AGGREGATED` | Summarises the selected input batch. Current SHAP bar, SHAP beeswarm, and tabular bar visualisers produce aggregated visual summaries when they aggregate local attributions. |
| `GLOBAL` | Represents a dataset, population, or model-wide result. The enum keeps this concept available, but built-in visualisers do not promote arbitrary batches to global outputs. |

The `AGGREGATED` scope distinction is intentional. A SHAP plotting API may call a bar or beeswarm
figure "global", but RAITAP only treats it as global when a first-class dataset, population, or
model-level contract proves that scope.

### `ScopeDefinitionStep`

Records where the scope was defined:

| Step | Meaning |
| --- | --- |
| `EXPLAINER_OUTPUT` | The explainer produced an artefact with this scope. |
| `VISUALISER_SUMMARY` | The visualiser changed the result scope by summarising another explanation artefact. |

For example, an attribution explainer produces local attributions with `EXPLAINER_OUTPUT`. A summary
visualiser consumes those local attributions and produces an aggregated figure with
`VISUALISER_SUMMARY`.

`VisualisationResult.scope` describes what the rendered figure represents. Reporting placement comes
from this rendered visualisation scope, not from legacy report-placement strings.

### `ExplanationOutputSpace`

Describes what attribution values are aligned to:

| Output space | Typical use |
| --- | --- |
| `INPUT_FEATURES` | Attributions aligned with input features, pixels, or tabular columns. |
| `INTERPRETABLE_FEATURES` | Attributions aligned with an interpretable feature representation. |
| `LAYER_ACTIVATION` | Attributions aligned with an internal model layer. |
| `IMAGE_SPATIAL_MAP` | CAM-style or spatial image maps that may need interpolation. |
| `TOKEN_SEQUENCE` | Token-level text attributions. |

Output-space inference relies on explicit input metadata and algorithm semantics. Shape alone is not
enough to decide whether a tensor is tabular, token, image, or time-series data.

**`LAYER_ACTIVATION` inference branch.** When `layer_path` is set and `MethodFamily.CAM` is not in the resolved method families, `infer_output_space` short-circuits to `LAYER_ACTIVATION` (skipping input-shape validation). CAM methods (`LayerGradCam`, `GuidedGradCam`) match the earlier CAM branch first and produce `IMAGE_SPATIAL_MAP` for image input, so they never reach this branch.

**`_needs_layer_resolution` prefix rule.** Any algorithm whose name starts with `Layer` (e.g. `LayerConductance`, `LayerIntegratedGradients`) or equals `GuidedGradCam` triggers layer resolution: the `layer_path` string in `constructor` is resolved to a live `nn.Module` before the Captum object is constructed. This happens automatically in `CaptumExplainer`; there is no extra flag to set.

### Sample identity vs display labels

RAITAP separates stable sample identity from display labels:

| Field | Purpose |
| --- | --- |
| `sample_ids` | Stable IDs from the data pipeline, when available. |
| `sample_display_names` | Optional labels used for plot titles. |

Display names are not stable identity. RAITAP must not infer dataset, population, or global
semantics from sample names shown in plots.

## Runtime flow

Transparency runs after the forward pass via `src/raitap/transparency/phase.py`
(`assess_transparency`). For each configured explainer:

1. `Explanation(config, name, model, data)` instantiates the explainer and its visualisers via hydra-zen.
2. `explainer.explain()` returns an `ExplanationResult` (for `AttributionOnlyExplainer`, after calling `compute_attributions()`).
3. `ExplanationResult.write_artifacts()` persists attributions and typed semantics to disk.
4. `ExplanationResult.visualise()` iterates the configured visualisers, validates each one against the explanation semantics, calls `visualise()`, and saves the figures.

Each explainer writes to its own subdirectory under the Hydra run folder. See {doc}`../../using-raitap/understanding-outputs` for the on-disk layout.

## Explanation-quality evaluation (Quantus, issue #341)

`src/raitap/transparency/evaluation/` grades attributions with
[Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus).
It is a separate sub-package, not a new explainer type: it grades an
already-computed `ExplanationResult`, it does not produce one.

```text
evaluation/
├── contracts.py                    # QuantusCategory, EvalRequirement, QuantusMetricSpec,
│                                    # EvaluationResult, EvaluationScore, SkippedMetric
├── semantics.py                    # EvaluationContext, resolve_metric (the requirement gate)
├── bridge.py                       # torch<->numpy + explain_func bridge (sole coupling point)
├── step.py                         # grade_explanations: the transparency-phase post-step
├── evaluators/
│   ├── base_evaluator.py           # BaseEvaluator(AdapterMixin, ABC)
│   ├── registration.py             # @transparency_evaluator decorator (family=None)
│   └── quantus_evaluator.py        # QuantusEvaluator + the 13-metric _REGISTRY
└── visualisers/
    └── score_visualisers.py        # ScoreBarVisualiser
```

### `BaseEvaluator` / `QuantusEvaluator`

`BaseEvaluator(AdapterMixin, ABC)` (`evaluators/base_evaluator.py`) declares
one abstract method: `evaluate(ctx: EvaluationContext, *, run_dir) ->
EvaluationResult`. It carries `algorithm_registry: ClassVar[Mapping[str,
QuantusMetricSpec]]`, set by the `@transparency_evaluator` decorator, the same
shape as `algorithm_registry` on `BaseExplainer` / `BaseAssessor`.

`QuantusEvaluator` (`evaluators/quantus_evaluator.py`) is the sole
implementation. Its `_REGISTRY` maps metric name to `QuantusMetricSpec`
across all 6 `QuantusCategory` values (faithfulness, complexity, robustness,
localisation, randomisation, axiomatic) - 13 entries. `evaluate()` resolves
each requested metric (`self.metrics` or, if unset, every registered metric)
via `resolve_metric`, runs the ones that resolve, and collects the rest as
`SkippedMetric`. `_run_metric` builds `{**spec.default_kwargs,
**self.constructor.get(key, {}), "disable_warnings": True}` as the Quantus
metric's constructor kwargs, then calls the metric with `{**resolved.call_kwargs,
**self.call}`, wrapped in `self._rethrow()` (inherited from `AdapterMixin`) so
library exceptions carry the raitap adapter-family context.

`@transparency_evaluator` (`evaluators/registration.py`) registers with
`family=None`, the same pattern `@visualisers.transparency` uses: no Hydra
config group, no schema dataclass, resolved purely by `_target_` nested under
`transparency.<name>.evaluation`. `extra` and `library` must be passed
explicitly (`extra="quantus"`, `library="quantus"`) since there is no family
default to fall back on.

`QuantusEvaluator.__init__` takes `call` and `raitap` via `**kwargs` rather
than named keyword-only params. Naming them explicitly would collide with the
`zen_meta={"call": {}, "raitap": {}}` hydra-zen attaches to every
family-less builder in `_register_core` - hydra-zen forbids a `zen_meta` key
that also exists in the target's own signature. `raitap.softmax` is the one
raitap-owned option read out of that block today.

### `EvalRequirement` gate

`EvaluationContext.available_requirements()` (`semantics.py`) computes what an
explanation can supply as a `frozenset[EvalRequirement]`:

| `EvalRequirement` | Set when |
| --- | --- |
| `ATTRIBUTIONS` | Always. |
| `MODEL` | `ctx.model is not None`. |
| `RE_EXPLAIN` | `ctx.explainer` is an `AttributionOnlyExplainer` (Captum, SHAP originate from this; `FullExplainer` subclasses do not). |
| `SEGMENTATION` | `ctx.masks is not None`. Always `None` today - no segmentation-mask provider exists yet (follow-up). |
| `BASELINE` | `ctx.baseline is not None` (the explainer's `BaselineRecord`, if any). |

`resolve_metric(metric, spec, ctx)` computes `spec.requires -
ctx.available_requirements()`. Empty means the metric runs: it returns a
`ResolvedMetric(spec, call_kwargs=ctx.gather(spec))`. Non-empty means it can't:
it returns a `SkippedMetric(metric, missing, message)` instead of raising.
This is a typed skip, not an error path - a config that requests
`pointing_game` on every explainer does not fail the run, it just always
skips until a segmentation-mask source exists. `QuantusEvaluator.evaluate`
branches on `isinstance(resolved, ResolvedMetric)` to sort the two outcomes
into `EvaluationResult.scores` / `.skipped`.

`EvaluationContext.gather(spec)` builds the Quantus call kwargs: always
`model`, `x_batch`, `y_batch`, `a_batch`, `device` (as `str`), `channel_first`,
`softmax`; adds `explain_func` when `RE_EXPLAIN in spec.requires` and
`s_batch` when `SEGMENTATION in spec.requires`.

### `bridge.py`: the sole coupling contract

`bridge.py` is deliberately the only file in `evaluation/` that imports from
`raitap.transparency.explainers` / `raitap.transparency.results`. Everything
else (`contracts.py`, `semantics.py`, `evaluators/`) works against the typed
dataclasses `bridge.py` hands back, never against `ExplanationResult` or
`AttributionOnlyExplainer` directly. Keep new coupling confined here.

Public fields it reads off `ExplanationResult`: `.inputs`, `.attributions`
(already CPU-detached float tensors per `ExplanationResult.__post_init__`),
`.semantics.output_space`, `.semantics.target`, `.call_kwargs`, `.name`,
`.adapter_target`, `.algorithm`, `.run_dir`, `.baseline`. The one method it
calls on an explainer is `AttributionOnlyExplainer.compute_attributions` - the
raw-tensor path, not `explain()` - because Quantus calls the wrapped
`explain_func` once per perturbation and `explain()` would write artifacts
every time.

Four functions:

- `to_quantus_arrays(result, *, target) -> QuantusArrays`: converts
  `.inputs` / `.attributions` to the numpy `(x_batch, y_batch, a_batch)`
  triple Quantus metrics expect.
- `resolve_target(result) -> int | list[int] | None`: reads the
  classification target from `call_kwargs["target"]`, falling back to
  `result.semantics.target`.
- `derive_channel_first(result) -> bool`: `True` for `IMAGE_SPATIAL_MAP`
  output space or any `BATCH_CHANNEL_HEIGHT_WIDTH` layout (plain gradient
  attributions on image input keep the input's NCHW layout even though their
  output space is `INPUT_FEATURES`, not `IMAGE_SPATIAL_MAP`).
- `explainer_to_explain_func(explainer, device) -> Callable`: wraps
  `compute_attributions` as a Quantus `explain_func(model, inputs, targets,
  **kwargs) -> np.ndarray`, tensorising Quantus's numpy args under
  `torch.enable_grad()` and returning a detached numpy array.

### Runtime wiring

`step.py::grade_explanations` is the transparency-phase post-step (called
from `phase.py::assess_transparency`, one call per configured explainer, after
its explanations are produced). It instantiates the evaluator from
`config.transparency[name].evaluation` (an `EvaluationConfig | None`) via
`hydra.utils.instantiate`, no-ops (`return []`) when that block is unset or
nothing was explained, pulls `model`/`device` off the `PreparedExplainer`'s
backend and puts the model in eval mode (`model.eval()`; several Quantus
MODEL-requiring metrics raise `AttributeError` on a training-mode module),
builds one `EvaluationContext` per `ExplanationResult`, and calls
`evaluator.evaluate(ctx, run_dir=result.run_dir)`. Results accumulate onto
`TransparencyOutput.evaluations` / `TransparencyPhaseResult.evaluations` (see
`phase.py`, `report.py`); `EvaluationResult.log(tracker)` logs each score's
`aggregate` under the `explanation_quality` metric prefix.

### Follow-ups (out of scope, tracked on issue #341)

- **Segmentation-mask provider**: `EvaluationContext.masks` is always `None`;
  no data-source wiring exists to fill `s_batch` for localisation metrics
  (`pointing_game`, `relevance_rank_accuracy`). Until one lands, both always
  skip via the `SEGMENTATION` requirement.
- **Stochastic re-explain**: `explainer_to_explain_func` re-runs
  `compute_attributions` once per Quantus perturbation with no seed pinning
  of its own. For a `global_rng`/`self_seeded` explainer (issue #339 seeding
  classification), robustness/randomisation metric values may not be
  bit-reproducible across runs even with a pinned run-level `seed`.

## Important files

- `src/raitap/transparency/contracts.py`: `ExplanationScope`, `ScopeDefinitionStep`, `ExplanationPayloadKind`, `ExplanationOutputSpace`, `MethodFamily`, `VisualisationContext`, `VisualSummarySpec`. Also defines `ExplainerAlgorithmSpec`, including the `requires: frozenset[Capability]` field for per-algorithm capability declarations.
- `src/raitap/transparency/results.py`: `ExplanationResult` (semantics, `write_artifacts`, `visualise`) and `VisualisationResult`.
- `src/raitap/transparency/factory.py`: the `Explanation` class and helpers that turn config into live explainer + visualiser instances.
- `src/raitap/transparency/explainers/base_explainer.py`: `BaseExplainer` + `AttributionOnlyExplainer`.
- `src/raitap/transparency/explainers/full_explainer.py`: `FullExplainer`.
- `src/raitap/transparency/visualisers/base_visualiser.py`: `BaseVisualiser` and the semantic-contract ClassVars.
- `src/raitap/transparency/evaluation/contracts.py`: `QuantusCategory`, `EvalRequirement`, `QuantusMetricSpec`, `EvaluationResult`, `EvaluationScore`, `SkippedMetric`.
- `src/raitap/transparency/evaluation/semantics.py`: `EvaluationContext`, `resolve_metric` (the requirement gate).
- `src/raitap/transparency/evaluation/bridge.py`: the sole coupling point into explainer internals.
- `src/raitap/transparency/evaluation/step.py`: `grade_explanations`, the transparency-phase post-step.
- `src/raitap/transparency/evaluation/evaluators/quantus_evaluator.py`: `QuantusEvaluator` + the metric `_REGISTRY`.

**Name resolution.** Bare class names in YAML `_target_` keys (e.g. `_target_: CaptumExplainer`) are resolved through the `@adapters.transparency` / `@visualisers.transparency` decorators and `raitap._adapters.lookup("transparency", name)`, not via the legacy class-kwarg path. To make a new class addressable by bare name, decorate it; that's the only requirement.
