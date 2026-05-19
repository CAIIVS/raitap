---
title: "Adding a task family"
description: "How to add a new task family (e.g. segmentation, seq2seq) to RAITAP — TaskKind enum, adapter declarations, scalar wrappers, ForwardOutput payload, per-phase dispatch, output spaces, per-element explanations, and the data + visualiser plumbing. Detection (issue #146) is the worked example."
myst:
  html_meta:
    "description": "How to add a new task family (e.g. segmentation, seq2seq) to RAITAP — TaskKind enum, adapter declarations, scalar wrappers, ForwardOutput payload, per-phase dispatch, output spaces, per-element explanations, and the data + visualiser plumbing. Detection (issue #146) is the worked example."
---

# Adding a task family

A **task family** (also "task kind") is the shape of the model's forward output: classification returns a `(batch, num_classes)` tensor, detection returns `list[dict[str, Tensor]]`, segmentation returns a per-pixel mask, seq2seq returns variable-length token sequences, regression returns a scalar-per-sample tensor. RAITAP's classification path was hard-coded for years; issue #146 generalised it to a `TaskKind` discriminator threaded through every pipeline phase.

This page covers the contributor workflow for adding a new family. Adding a new *adapter* to an existing family is much smaller — see {doc}`adding-an-adapter`.

## When to add a task family

Trigger: the model's forward output is not a scalar-per-sample tensor and the *user-facing semantics* of explanation / metrics / robustness differ from classification.

- If the output can be reduced to a scalar via a thin wrapper without changing what the user sees (e.g. you want "explain the score of box #0" rather than "explain each box individually"), you may not need a new family — just write a wrapper and reuse `TaskKind.classification`. The pattern is `ScalarDetectionWrapper` in `src/raitap/models/task_wrappers.py`; you can take the same approach for any non-scalar output.
- Detection needed a new family because per-box explanation semantics differ from per-class: K boxes per sample, each with its own attribution map, its own ground-truth correspondence, its own faithfulness measurement. A scalar reduction throws that structure away.

The rest of this page assumes you concluded a new family is justified.

## Extension pattern

Thirteen steps. Each cites the file you edit (paths are `file:line` where useful) and shows the detection version of the change.

### 1. Add a `TaskKind` enum member

`src/raitap/types.py:35` defines the `TaskKind` `StrEnum`. Member name and value must be identical (lowercase) so OmegaConf can round-trip them through YAML.

```python
class TaskKind(StrEnum):
    classification = "classification"
    detection = "detection"
    segmentation = "segmentation"
    seq2seq = "seq2seq"
    regression = "regression"
```

### 2. Declare `supported_tasks` on adapters

`src/raitap/_adapters.py:120` defines the default on `AdapterMixin`:

```python
supported_tasks: ClassVar[frozenset[TaskKind]] = frozenset({TaskKind.classification})
```

Every adapter (explainer / assessor / metric / visualiser) that supports the new family overrides this ClassVar. Adapters that *don't* support it stay on the default — legacy classification adapters need no edit.

```python
class DetectionImageVisualiser(BaseVisualiser):
    supported_tasks: ClassVar[frozenset[TaskKind]] = frozenset({TaskKind.detection})
```

See `src/raitap/transparency/visualisers/detection_image_visualiser.py:61`.

### 3. Decide whether you need a `ScalarXWrapper`

If your new family also wants to run the existing scalar-output explainers (Captum, SHAP, gradient-based attacks) alongside the per-element ones, ship a wrapper that reduces the structured output to a scalar tensor. Pattern: `ScalarDetectionWrapper` in `src/raitap/models/task_wrappers.py` (the `nn.Module` wrapper) plus `DetectionTarget` (the reducer, with modes `class_score` / `objectness` / `bbox_l2` / `reference_match`).

```python
class ScalarDetectionWrapper(nn.Module):
    def __init__(self, model: nn.Module, *, target: DetectionTarget) -> None:
        super().__init__()
        self.model = model
        self.target = target

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.model(inputs)
        ...
        return torch.stack(per_sample).reshape(-1, 1)
```

The reducer's `reference_match` mode is the one used by the per-box detection explain phase; it anchors the scalar to a specific reference box across perturbations rather than drifting with output-list reordering. Other modes exist for debugging.

### 4. Backend auto-detection (or explicit opt-in)

`src/raitap/models/backend.py:106` houses `_is_torchvision_detection_model` — a duck-typed check that runs at `TorchBackend.__init__` time. Either extend this helper for your model class, or rely on the explicit `task_kind=` kwarg on `TorchBackend`:

```python
class TorchBackend(ModelBackend):
    def __init__(
        self,
        model: nn.Module,
        ...,
        task_kind: TaskKind | None = None,
    ) -> None:
        ...
        self._task_kind = task_kind if task_kind is not None else self._infer_task_kind(model)
```

The `task_kind` property on `ModelBackend` (`src/raitap/models/backend.py:144`) is the single source of truth the pipeline reads.

### 5. Extend `ForwardOutput`

`src/raitap/pipeline/outputs.py:31` is the typed dataclass returned by `forward_pass`. Add a new `Optional[<your_payload_type>]` field and extend the `__post_init__` invariant for the new task↔payload pairing:

```python
@dataclass(frozen=True)
class ForwardOutput:
    task_kind: TaskKind
    batch_size: int
    predictions_tensor: torch.Tensor | None = None
    detection_predictions: list[dict[str, torch.Tensor]] | None = None
    # add your_kind_predictions: <your type> | None = None

    def __post_init__(self) -> None:
        if self.task_kind is TaskKind.classification and self.predictions_tensor is None:
            raise ValueError("ForwardOutput(task_kind=classification) requires predictions_tensor.")
        if self.task_kind is TaskKind.detection and self.detection_predictions is None:
            raise ValueError("ForwardOutput(task_kind=detection) requires detection_predictions.")
        # add the equivalent check for your family
```

`batch_size` stays task-agnostic so reporting + UI callers don't branch.

### 6. Branch in `forward_pass`

`src/raitap/pipeline/phases/forward_pass.py:103` reads `backend.task_kind` and writes the typed payload:

```python
task_kind = getattr(backend, "task_kind", TaskKind.classification)

if task_kind is TaskKind.detection:
    detection_predictions: list[dict[str, torch.Tensor]] = []
    ...
    return ForwardOutput(
        task_kind=TaskKind.detection,
        batch_size=len(detection_predictions),
        detection_predictions=detection_predictions,
    )
```

Add the equivalent branch for your kind.

### 7. Branch in the downstream phases + add a typed metrics config

Three phases dispatch on `forward_output.task_kind`:

- `src/raitap/pipeline/phases/evaluate_metrics.py:49` — pass the structured payload straight to the family-specific metric computer.
- `src/raitap/pipeline/phases/assess_transparency.py:71` — route detection to `_assess_transparency_detection` (see step 9).
- `src/raitap/pipeline/phases/assess_robustness.py:78` — early-return for any non-classification kind until Phase 4 lands (see cross-cutting concerns).

For metrics, add a typed schema subclass that defaults everything so classification configs stay untouched. Mirror `DetectionMetricsConfig` at `src/raitap/configs/schema.py:227`.

### 8. Add an `ExplanationOutputSpace` member + wire semantics

`src/raitap/transparency/contracts.py:49` defines `ExplanationOutputSpace`. Add a new member:

```python
class ExplanationOutputSpace(StrEnum):
    ...
    DETECTION_BOXES = "detection_boxes"
    # YOUR_SPACE = "your_space"
```

Then in `src/raitap/transparency/semantics.py`:

- `_candidate_output_spaces` (line 330) — early-return your space when `task_kind` matches.
- `infer_output_space` (line 154) — early-return your space when `task_kind` matches.

Detection example (`semantics.py:167`):

```python
if task_kind is TaskKind.detection:
    return ExplanationOutputSpace.DETECTION_BOXES
```

The pre-flight check in `src/raitap/transparency/factory.py:124` resolves the backend *before* the semantic compat check and passes `task_kind=backend.task_kind` through, so any visualiser whose `supported_tasks` / `supported_output_spaces` excludes the new family fails fast with a clear message.

### 9. Per-element explanations (optional but typical)

If the new family needs more than one explanation per sample (detection: one per box; segmentation: arguably one per instance mask; seq2seq: per-token), write:

- A new `src/raitap/pipeline/phases/explain_X.py` mirroring `explain_detection.py` (per-box K-loop).
- An `_assess_transparency_X` helper in `assess_transparency.py` that calls it (mirror `_assess_transparency_detection`).
- A per-element metadata dataclass on `transparency/contracts.py` (mirror `DetectionBox` at line 174).
- A field on `ExplanationResult` carrying that metadata (mirror `detection_box: DetectionBox | None = None` at `src/raitap/transparency/results.py:147`) and a `original_sample_index: int | None = None` short-circuit in `render_visualisation_for_scope` so visualisers only render the original-sample tile when the scope matches (`results.py:348`).

### 10. Write a visualiser

```python
@register_transparency_visualiser(registry_name="detection_image")
class DetectionImageVisualiser(BaseVisualiser):
    supported_output_spaces: ClassVar[frozenset[ExplanationOutputSpace]] = frozenset(
        {ExplanationOutputSpace.DETECTION_BOXES}
    )
    supported_tasks: ClassVar[frozenset[TaskKind]] = frozenset({TaskKind.detection})
```

See `src/raitap/transparency/visualisers/detection_image_visualiser.py:35`. It's re-exported in `src/raitap/transparency/visualisers/__init__.py` so users can pick it via the stable string name.

The general pattern for adapter registration is identical to any other family — see {doc}`adding-an-adapter`.

### 11. Data layer

If labels for the new family aren't a simple tensor, add a `_load_X_labels` codepath in `src/raitap/data/data.py` gated by the `data.labels.kind` discriminator. Detection precedent:

```python
labels_kind = _get_optional_config_value(labels_cfg, "kind")
...
if labels_kind == "detection":
    self.labels = self._load_detection_labels(cfg)
```

(`src/raitap/data/data.py:59`, `_load_detection_labels` at line 216.) The discriminator field is `kind: str | None = None` on `LabelsConfig` at `src/raitap/configs/schema.py:79`.

> **Discrepancy with the plan.** The plan referenced a `LabelKind` enum in `src/raitap/data/types.py`. As of writing there is no such enum — `data.labels.kind` is a free-form `str` checked against literal `"detection"`. If you want a typed enum, introduce one now; otherwise follow the existing string-discriminator pattern.

### 12. Tests

Three layers:

- **Unit** — one test file per new component (wrapper, target, visualiser, label loader, metric).
- **Per-phase** — each pipeline phase that dispatches on `task_kind` needs a test asserting it routes correctly when handed a `ForwardOutput(task_kind=<new>)`. Existing examples: `src/raitap/pipeline/phases/tests/test_explain_detection.py`, `src/raitap/metrics/tests/test_detection_metrics.py`, `src/raitap/data/tests/test_detection_labels.py`.
- **Integration / E2E** — wire the whole pipeline against a real model + small dataset (the detection contributor config below is the artefact).

Backward compatibility is part of the contract: every classification-only path must stay bit-for-bit identical when the new family isn't selected. Add an explicit assertion to one classification E2E test that touching the new family's code didn't perturb it.

### 13. Docs

Three pages:

- **Configuration knobs** for the module — mirror the "Detection knobs" section at `docs/modules/transparency/configuration.md:187`.
- **Visualiser entry** — mirror the "Detection" / "DetectionImageVisualiser" section at `docs/modules/transparency/visualisers.md:237`.
- **Contributor config** — a runnable end-to-end YAML under `contributor-configs/<your_family>-<dataset>/`. See `contributor-configs/fasterrcnn-udacity/` as the detection example.

## Worked example: detection (issue #146)

The 24 locked design decisions (D1–D24) made during the brainstorm live in `docs/superpowers/specs/2026-05-17-detection-image-visualiser-design.md`. The detection family touched exactly the files in the recipe above:

| Step | File(s) | Why |
|---|---|---|
| 1 | `src/raitap/types.py` | `TaskKind.detection` enum member |
| 2 | `src/raitap/_adapters.py` | `supported_tasks` ClassVar default on `AdapterMixin` |
| 3 | `src/raitap/models/task_wrappers.py` | `ScalarDetectionWrapper` + `DetectionTarget` (4 reducer modes) |
| 4 | `src/raitap/models/backend.py` | `task_kind` property, `_is_torchvision_detection_model` auto-detect, explicit `task_kind=` kwarg on `TorchBackend` |
| 5 | `src/raitap/pipeline/outputs.py` | `ForwardOutput.detection_predictions` + `__post_init__` invariant |
| 6 | `src/raitap/pipeline/phases/forward_pass.py` | task-kind branch collecting `list[dict[str, Tensor]]` |
| 7 | `src/raitap/pipeline/phases/evaluate_metrics.py`, `assess_transparency.py`, `assess_robustness.py`; `src/raitap/configs/schema.py` (`DetectionMetricsConfig`) | per-phase dispatch + typed metrics schema |
| 8 | `src/raitap/transparency/contracts.py`, `semantics.py`, `factory.py` | `DETECTION_BOXES` output space, candidate / inference helpers, task-aware pre-flight |
| 9 | `src/raitap/pipeline/phases/explain_detection.py`, `assess_transparency.py::_assess_transparency_detection`, `transparency/contracts.py::DetectionBox`, `transparency/results.py::ExplanationResult.{detection_box,original_sample_index}` | per-box K-loop + metadata threading |
| 10 | `src/raitap/transparency/visualisers/detection_image_visualiser.py`, `visualisers/__init__.py` | `DetectionImageVisualiser` registered as `detection_image` |
| 11 | `src/raitap/data/data.py::_load_detection_labels`, `src/raitap/configs/schema.py::LabelsConfig.kind` | `data.labels.kind == "detection"` discriminator |
| 12 | `src/raitap/pipeline/phases/tests/test_explain_detection.py`, `src/raitap/metrics/tests/test_detection_metrics.py`, `src/raitap/data/tests/test_detection_labels.py` | per-component + per-phase coverage |
| 13 | `docs/modules/transparency/configuration.md`, `docs/modules/transparency/visualisers.md`, `contributor-configs/fasterrcnn-udacity/` | user docs + contributor config |

## Cross-cutting concerns

- **Metrics schema.** mAP-style detection metrics use the typed-schema pattern from main (`DetectionMetricsConfig` at `src/raitap/configs/schema.py:227`) which defaults everything. Classification configs are untouched. Apply the same shape to your family.
- **Reporting.** The K-results-per-sample layout (decision D15 in the spec) is currently still deferred — the per-sample-grouped report section is planned but not in main. New per-element families inherit that gap until D15 lands.
- **Robustness.** Per-family robustness is a separate deliverable. `assess_robustness` short-circuits on any non-classification `task_kind` at `src/raitap/pipeline/phases/assess_robustness.py:78`. For detection specifically, Phase 4 (`DetectionAdversarialLoss`) is the planned follow-up.
- **Backward compatibility.** Every classification-only path stays bit-for-bit identical when the new family isn't selected. This is part of the test contract — assert it explicitly in integration.
- **Visualiser pre-flight.** `transparency/factory.py` resolves the backend *before* the semantic compat check, so a visualiser whose `supported_tasks` excludes the active task_kind raises during config validation, not deep in the K-loop. Keep that ordering when adding family-specific visualisers.
