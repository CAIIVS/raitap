---
title: "Contributing to the transparency module"
description: "Transparency-specific architecture: the three-level explainer hierarchy, the BaseVisualiser semantic contract, and the ExplanationResult typed semantics."
myst:
  html_meta:
    "description": "Transparency-specific architecture: the three-level explainer hierarchy, the BaseVisualiser semantic contract, and the ExplanationResult typed semantics."
---

# Contributing to the transparency module

Generic adapter mechanics are covered elsewhere:

- New library wrapper: {doc}`adding-an-adapter`.
- New algorithm on an existing wrapper: {doc}`adding-an-algorithm`.
- New top-level module: {doc}`adding-a-module`.

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
  live on `ExplainerSemanticsHints.requires`. Never subclass directly.
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
| `ExplainerSemanticsHints.baseline_default` | per-algorithm `algorithm_registry` entry | Per-**algorithm** implicit default mode (`BaselineMode.ZERO` / `INPUT_BATCH`) used when the kwarg is omitted; `None` when the algorithm takes no baseline. |
| `ExplainerSemanticsHints.baseline_cardinality` | per-algorithm `algorithm_registry` entry | `BaselineCardinality.SINGLE` (one broadcast reference, e.g. IG) or `SET` (a sample distribution, e.g. SHAP). Used to *warn* on a mismatched `raitap.baseline` (never to reshape it); `None` skips the check. |

Capture happens once at the `AttributionOnlyExplainer.explain` chokepoint via `build_baseline_record`
(`transparency/baselines.py`), which resolves the `BaselineMode` (`configured` / `user_tensor` /
`zero` / `input_batch`), hashes the tensor, and renders an image preview. It is wrapped so a
render/hash failure degrades to no baseline rather than discarding attributions.
See [Adding an algorithm](adding-an-algorithm.md).

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
see {doc}`adding-an-adapter`. The bullets above are the transparency-specific additions on top of
that scaffolding.

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

Each explainer writes to its own subdirectory under the Hydra run folder. See {doc}`../using-raitap/understanding-outputs` for the on-disk layout.

## Important files

- `src/raitap/transparency/contracts.py`: `ExplanationScope`, `ScopeDefinitionStep`, `ExplanationPayloadKind`, `ExplanationOutputSpace`, `MethodFamily`, `VisualisationContext`, `VisualSummarySpec`. Also defines `ExplainerSemanticsHints`, including the `requires: frozenset[Capability]` field for per-algorithm capability declarations.
- `src/raitap/transparency/results.py`: `ExplanationResult` (semantics, `write_artifacts`, `visualise`) and `VisualisationResult`.
- `src/raitap/transparency/factory.py`: the `Explanation` class and helpers that turn config into live explainer + visualiser instances.
- `src/raitap/transparency/explainers/base_explainer.py`: `BaseExplainer` + `AttributionOnlyExplainer`.
- `src/raitap/transparency/explainers/full_explainer.py`: `FullExplainer`.
- `src/raitap/transparency/visualisers/base_visualiser.py`: `BaseVisualiser` and the semantic-contract ClassVars.

**Name resolution.** Bare class names in YAML `_target_` keys (e.g. `_target_: CaptumExplainer`) are resolved through the `@adapters.transparency` / `@visualisers.transparency` decorators and `raitap._adapters.lookup("transparency", name)`, not via the legacy class-kwarg path. To make a new class addressable by bare name, decorate it; that's the only requirement.
