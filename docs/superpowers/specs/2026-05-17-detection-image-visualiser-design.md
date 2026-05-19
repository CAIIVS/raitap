# Detection Image Visualiser + Per-Box Explanation Pipeline — Design Spec

**Issue:** #146 Phase 3
**Branch (when implemented):** TBD (off `main`, after PR #176 merges)
**Status:** design locked, ready for plan
**Effort estimate:** ~3 weeks single engineer (option C: explanations + visualiser + minimal metrics plumbing; defer detection metric reporting). Revised upward from ~2 weeks after second design-review pass (typed `ForwardOutput`, dedicated detection-label loader, factory reorder, target normalisation).

## Goal

For any detection model (Faster R-CNN / RetinaNet / SSD detected via `backend.task_kind == detection`), produce **one faithful `ExplanationResult` per detected box**. Each result targets that exact box's class score using an IoU-anchored reference, renders as a single figure (original image with the box highlighted + per-pixel attribution heatmap + class/score caption), and persists alongside its peers under a per-sample / per-box directory.

Detection metrics (`DetectionMetrics`, already registered in the ZenStore) gain end-to-end plumbing so a Faster R-CNN run can compute mAP without crashing the pipeline. Report rendering of detection metrics is deferred.

## Tech Stack

- Python 3.13, PyTorch, torchvision (detection models), torchmetrics (`mean_ap.MeanAveragePrecision`)
- Hydra / hydra-zen / OmegaConf for config
- pytest, pyright, ruff, `uv run`
- Existing RAITAP foundation from PR #176 (`TaskKind`, `supported_tasks` on `AdapterMixin`, `DetectionTarget`, `ScalarDetectionWrapper`, `DETECTION_BOXES` output space, `infer_output_space(task_kind=…)`)

---

## Locked decisions

| # | Decision | Source |
|---|---|---|
| D1 | Per-box attribution strategy: one attribution per detected box (faithful) | brainstorm |
| D2 | Box selection filter: `score >= score_threshold`, then top `max_boxes` by score descending | brainstorm |
| D3 | Orchestration: pipeline-level K-loop in a new phase | brainstorm |
| D4 | Result granularity: one `ExplanationResult` per box (K results per detection sample) | brainstorm |
| D5 | Reporting layout: R1 (sequential, sample_id chip groups visually); no new section type this phase | brainstorm |
| D6 | DetectionTarget faithful mode: new `"reference_match"` mode (IoU + label match against a fixed reference box) | review correction |
| D7 | `DetectionBox` carries both `display_index` (rank in filtered set 0..K-1) and `raw_index` (index in clean forward output) | review correction |
| D8 | Task-aware factory pre-flight: `explainer_capability(task_kind=…)` + `_candidate_output_spaces(task_kind=…)` so DETECTION_BOXES becomes a valid candidate when backend declares detection | review correction |
| D9 | Thread `backend.task_kind` through `AttributionOnlyExplainer.explain()` into `infer_output_space()` (the DETECTION branch added in PR #176 is otherwise dead code) | review correction |
| D10 | Per-box run_dir: `transparency/{explainer}/sample_{i}/box_{raw_index}/` constructed by the detection phase | review correction |
| D11 | Config placement: detection knobs under `transparency.<explainer>.raitap.detection.{score_threshold, max_boxes, iou_threshold}` (no schema change — existing `raitap: dict[str, Any]` field already accepts arbitrary keys) | review correction |
| D12 | `ExplanationResult.original_sample_index: int \| None = None` field; `render_visualisation_for_scope()` becomes detection-aware (skip when `sample_index != original_sample_index`; use stored tensors verbatim without slicing) | review correction |
| D13 | Empty-detections handling: skip + warn (no placeholder `ExplanationResult` — `attributions=None` is incompatible with the required `torch.Tensor` field at `results.py:127`) | review correction |
| D14 | COCO label-name lookup: deferred — `label_name=None` default; dataset config may populate later via a separate plan | brainstorm |
| D15 | Sample-level grouped report section (overview tile per sample): deferred to a follow-up PR after R1 lands | brainstorm |
| D16 | Multi-model detector support beyond torchvision (DETR / YOLO): deferred | brainstorm |
| D17′ | Forward-pass preserves `list[dict]` for detection: new helper `extract_detection_predictions()` alongside the existing `extract_primary_tensor()`; called when `backend.task_kind == detection` | metrics review |
| D18′ | `evaluate_metrics` gains detection branch dispatching on `backend.task_kind`; consumes `list[dict]` predictions + dataset detection targets via `DetectionMetrics.update()` | metrics review |
| D19 | Detection metric reporting (mAP table, class-AP table) in PDF / HTML: deferred to a follow-up PR | metrics review |
| D20 | Docs: one-liner under `docs/modules/metrics/` showing canonical `metrics: { _target_: DetectionMetrics }` example; plus a new `docs/modules/transparency/visualisers.md` entry for `DetectionImageVisualiser` | metrics review |
| D21 | Detection phase normalises explainer `call.target = 0` (wrapper exposes one scalar channel). `auto_pred` is rejected explicitly with `RaitapError` — argmax over a 1-channel output always returns 0, masking real config errors. Documented in `docs/modules/transparency/configuration.md`. | second review |
| D22 | Detection labels load via a dedicated `Data._load_detection_labels()` codepath returning `list[dict[str, Tensor]]` (per-sample `boxes: (M_i, 4)` xyxy + `labels: (M_i,)` int64). Existing `_load_labels()` stays classification-only. Discriminator in YAML: `data.labels.kind: detection`. | second review |
| D23 | Replace `RunOutputs.forward_output: torch.Tensor` (`outputs.py:28`) with a typed `ForwardOutput` dataclass carrying `predictions_tensor: torch.Tensor \| None`, `detection_predictions: list[dict[str, Tensor]] \| None`, `batch_size: int`, `task_kind: TaskKind`. Six downstream callsites migrate from `.ndim`/`.shape[0]`/`.shape[1]` to `.batch_size` + `.task_kind` branches: `builder.py:1260`, `prediction_summaries.py:30`, `assess_robustness.py:38`, `evaluate_metrics.py:41-44`, plus the two phases consuming it. | second review (5b) |
| D24 | Detection explain phase READS the pre-computed boxes from `ForwardOutput.detection_predictions` rather than re-running `backend(inputs)` per sample. Single forward pass per run; metrics and explanations anchor to the same clean outputs (consistency); avoids duplicate inference cost. | second review |

---

## Architecture

Three layers, mirroring the foundation in PR #176.

### Layer 1 — typed model (contracts)

**New `DetectionBox` frozen dataclass** in `src/raitap/transparency/contracts.py`:

```python
@dataclass(frozen=True)
class DetectionBox:
    display_index: int                                # rank in the filtered set (0..K-1)
    raw_index: int                                    # index in the clean forward pass's output list
    xyxy: tuple[float, float, float, float]           # bbox coords in input pixel space
    score: float                                      # detection confidence
    label_index: int                                  # torchvision class id
    label_name: str | None = None                     # human-readable name if dataset provides one
```

**New field on `ExplanationResult`** in `src/raitap/transparency/results.py`:

```python
detection_box: DetectionBox | None = None             # populated by detection phase; None for classification
original_sample_index: int | None = None              # which global sample this result belongs to (D12)
```

Persisted in `metadata.json` by `write_artifacts()`. Default `None` → classification path untouched.

**New field on `VisualisationContext`** in `src/raitap/transparency/contracts.py`:

```python
detection_box: DetectionBox | None = None
```

Propagated by `ExplanationResult.visualise()` and by `render_visualisation_for_scope()` when constructing the context.

### Layer 2 — pipeline detection phase

**New module** `src/raitap/pipeline/phases/explain_detection.py` (sibling of `assess_transparency.py`). The existing `assess_transparency` phase dispatches: when `backend.task_kind == detection`, delegate to `explain_detection`; else current behaviour unchanged.

Per detection sample `i`:

1. **Read pre-computed predictions** (D24). `predictions_i = forward_output.detection_predictions[i]` → `dict[str, Tensor]` with `boxes`, `scores`, `labels`. No second forward pass — the clean predictions were already produced by the upstream `forward_pass` phase and persist on `ForwardOutput`. Read knobs from the explainer's `raitap.detection` config block (defaults: `score_threshold=0.5`, `max_boxes=5`, `iou_threshold=0.5`).
2. **Filter — correct raw-index path** (D2 + A1 fix). Mask + indirect index so the resulting indices reference the ORIGINAL detector output, not the filtered subset:

   ```python
   mask = predictions_i["scores"] >= score_threshold
   raw_candidates = torch.nonzero(mask, as_tuple=False).flatten()
   order = predictions_i["scores"][raw_candidates].argsort(descending=True)
   top_k_raw_indices = raw_candidates[order[:max_boxes]]   # raw indices into clean output
   K = top_k_raw_indices.numel()
   ```

   `display_index = j` (position in `top_k_raw_indices`); `raw_index = int(top_k_raw_indices[j])`. If `K == 0`: log warn (sample_id + threshold + `max_score = predictions_i["scores"].max().item() if scores.numel() else None`), continue to next sample, emit no results.
3. **Per-box loop.** For each `(display_index, raw_index)` pair:
   - Build reference: `reference_xyxy = predictions[0]["boxes"][raw_index].tolist()`, `reference_label = int(predictions[0]["labels"][raw_index])`, `score = float(predictions[0]["scores"][raw_index])`.
   - Wrap detector: `target = DetectionTarget(mode="reference_match", reference_xyxy=reference_xyxy, reference_label=reference_label, iou_threshold=iou_threshold)`; `wrapped = ScalarDetectionWrapper(backend.as_model_for_explanation(), target=target)`.
   - Construct per-box run_dir: `<base>/sample_{i}/box_{raw_index}/`.
   - Call `explainer.explain(wrapped, inputs[i:i+1], backend=backend, run_dir=<per-box-path>, ...)` → `ExplanationResult`.
   - Attach `detection_box=DetectionBox(display_index, raw_index, xyxy, score, label_index, label_name=None)` and `original_sample_index=i` on the returned result.
   - Yield into the result stream.

`infer_output_space(task_kind=TaskKind.detection)` tags each result's semantics with `output_space=DETECTION_BOXES` — this is automatic because of D9 (explainer threads `backend.task_kind`).

### Layer 3 — visualiser

**New** `src/raitap/transparency/visualisers/detection_image_visualiser.py`:

```python
@register_transparency_visualiser(registry_name="DetectionImage")
class DetectionImageVisualiser(BaseVisualiser):
    supported_payload_kinds = frozenset({ExplanationPayloadKind.ATTRIBUTIONS})
    supported_output_spaces = frozenset({ExplanationOutputSpace.DETECTION_BOXES})
    supported_scopes = frozenset({ExplanationScope.LOCAL})
    supported_method_families = frozenset({
        MethodFamily.GRADIENT,
        MethodFamily.PERTURBATION,
        MethodFamily.SHAPLEY,
        MethodFamily.CAM,
        MethodFamily.MODEL_AGNOSTIC,
        MethodFamily.SURROGATE,
    })
    embeds_original_input = True
    supported_tasks = frozenset({TaskKind.detection})   # opt-in to detection task
```

`visualise(attributions, inputs, *, context, **kwargs)` produces a single-panel `Figure`:
- Original image (denormalised if metadata provides mean/std; else assumed in `[0, 1]`).
- Rectangle outline at `context.detection_box.xyxy` (consistent colour per `label_index`).
- Heatmap overlay from `attributions` (per-pixel, optional sign-handling reusing existing Captum-style colour mapping).
- Title: `f"{label_name or f'class {label_index}'}: {score:.2f}"` + chip showing `display_index/K` so reader knows ordinal.
- Colorbar consistent with the other RAITAP image visualisers.

Reuses helpers from `CaptumImageVisualiser` where possible (denormalisation, colorbar, sign mapping).

### Compatibility plumbing (the 7 review fixes)

**Fix D6 — Faithful `DetectionTarget.reference_match` mode** in `src/raitap/models/task_wrappers.py`:

```python
DetectionTarget(
    mode="reference_match",
    reference_xyxy=(x1, y1, x2, y2),
    reference_label=int,
    iou_threshold=0.5,    # default
)
```

On `__call__(list[dict])`:
1. For each predicted box in each sample, compute IoU with `reference_xyxy` AND check `labels == reference_label`.
2. Among matches with `IoU >= iou_threshold`, return the score of the highest-IoU prediction.
3. If no match: return `torch.tensor(0.0)` on the appropriate device.

`box_idx` mode remains for raw-output debugging — docstring clarifies it indexes the detector's output list as-emitted (NOT a stable semantic reference). New `Literal["class_score", "objectness", "bbox_l2", "reference_match"]` mode union.

**Fix D8 — Task-aware candidate output spaces** in `src/raitap/transparency/semantics.py`:

```python
def _candidate_output_spaces(
    method_families: frozenset[MethodFamily],
    task_kind: TaskKind | None = None,
) -> frozenset[ExplanationOutputSpace]:
    if task_kind is TaskKind.detection:
        return frozenset({ExplanationOutputSpace.DETECTION_BOXES})
    # ... existing CAM / non-CAM branches unchanged

def explainer_capability(
    explainer: object, *, task_kind: TaskKind | None = None
) -> ExplainerCapability: ...
```

**Factory plumbing** in `src/raitap/transparency/factory.py` requires a **call-order swap** (A2 fix). Current order at lines 178/183 runs the semantic compat check before `_require_model_backend(model)` resolves the backend — `task_kind` isn't available yet. Reorder:

```python
# Before (current main):
#     check_explainer_visualiser_semantic_compat(explainer, explainer_target, visualisers)   # line 178
#     backend = _require_model_backend(model)                                                  # line 183

# After:
backend = _require_model_backend(model)
check_explainer_visualiser_semantic_compat(
    explainer,
    explainer_target,
    visualisers,
    task_kind=backend.task_kind,                          # new kwarg threaded through
)
```

Inside `check_explainer_visualiser_semantic_compat`:

```python
capability = explainer_capability(explainer, task_kind=task_kind)
```

Pre-flight then passes a `DetectionImageVisualiser` (whose `supported_output_spaces == {DETECTION_BOXES}`) when (and only when) the backend declares detection.

**Fix D9 — Thread task_kind through explain()** in `src/raitap/transparency/explainers/base_explainer.py:138`:

```python
output_space = infer_output_space(
    input_spec=input_spec,
    attributions=attributions,
    explainer=self,
    method_families=method_families,
    layer_path=_layer_path_for_explainer(self),
    task_kind=getattr(backend, "task_kind", None),
)
```

Without this patch the DETECTION branch added in PR #176 is dead code.

**Fix D10 — Per-box run_dir.** Detection phase constructs `transparency/{explainer_name}/sample_{i}/box_{raw_index}/` and overrides `run_dir=` per `explainer.explain(...)` call. Factory's default `resolve_run_dir(config, subdir=f"transparency/{explainer_name}")` stays untouched for classification.

**Fix D11 — Config placement.** Knobs live under the existing `raitap: dict[str, Any]` bag on `TransparencyConfig`:

```yaml
transparency:
  my_ig_explainer:
    _target_: CaptumExplainer
    algorithm: IntegratedGradients
    call:
      target: 0                              # D21 — wrapper exposes one scalar channel
    raitap:
      detection:
        score_threshold: 0.5
        max_boxes: 5
        iou_threshold: 0.5
```

Detection phase reads `raitap_kwargs.get("detection", {})`. No schema change.

**Fix D21 — target normalisation.** Detection phase intercepts `call.target` before invoking the explainer:

```python
target = call_kwargs.get("target", 0)
if target == "auto_pred":
    raise RaitapError(
        "config.transparency.<explainer>.call.target=auto_pred is not supported for "
        "detection tasks: the ScalarDetectionWrapper exposes a single scalar channel, "
        "so argmax over it always returns 0. Set call.target=0 explicitly."
    )
if target != 0:
    raitap_log.warn(
        f"Overriding call.target={target!r} to 0 for detection task (wrapper exposes "
        "a single scalar channel)."
    )
call_kwargs["target"] = 0
```

Documented in `docs/modules/transparency/configuration.md`.

**Fix D22 — Detection labels loader.** New method on `Data` in `src/raitap/data/data.py`:

```python
def _load_detection_labels(self, cfg: AppConfig) -> list[dict[str, torch.Tensor]] | None:
    """Load per-sample detection targets (boxes + labels per sample).

    Expected dataset shape on disk (one of):
      - JSON/CSV with rows {sample_id, x1, y1, x2, y2, label} → grouped by sample_id.
      - One JSON file per sample with {"boxes": [[…]], "labels": […]}.

    Returns list of length N, dict per sample with:
      "boxes": (M_i, 4) float32 xyxy
      "labels": (M_i,) int64
    """
```

Discriminator in YAML:

```yaml
data:
  labels:
    source: udacityselfdriving_boxes.json
    kind: detection                          # new — discriminates loader codepath
```

Validation: rejects samples whose box count mismatches the number of label entries; rejects labels outside `[0, num_classes)` when `num_classes` is known.

**Fix D23 — Typed `ForwardOutput` dataclass.** Replace `RunOutputs.forward_output: torch.Tensor` with a typed dataclass living in `src/raitap/pipeline/outputs.py`:

```python
@dataclass(frozen=True)
class ForwardOutput:
    task_kind: TaskKind
    batch_size: int
    predictions_tensor: torch.Tensor | None = None              # classification: (N, num_classes); detection: None
    detection_predictions: list[dict[str, torch.Tensor]] | None = None   # detection: length-N list; classification: None

    def __post_init__(self) -> None:
        if self.task_kind is TaskKind.classification and self.predictions_tensor is None:
            raise ValueError("ForwardOutput(classification) requires predictions_tensor.")
        if self.task_kind is TaskKind.detection and self.detection_predictions is None:
            raise ValueError("ForwardOutput(detection) requires detection_predictions.")

@dataclass(frozen=True)
class RunOutputs:
    ...
    forward_output: ForwardOutput              # was: torch.Tensor
    ...
```

Six callsites migrate:

| File:line | Change |
|---|---|
| `pipeline/phases/forward_pass.py` | construct `ForwardOutput(...)` instead of returning raw tensor |
| `pipeline/phases/prediction_summaries.py:30` | guard: only run argmax/softmax when `task_kind == classification`; detection skips (no per-class confidence concept) |
| `pipeline/phases/evaluate_metrics.py:41-44` | branch on `task_kind`; detection passes `detection_predictions` to `DetectionMetrics.update()` |
| `pipeline/phases/assess_robustness.py:38` | guard: existing assertion `ndim != 2 or shape[1] < 2` becomes a `task_kind == classification` check |
| `reporting/builder.py:1260` | replace `outputs.forward_output.ndim > 0` + `.shape[0]` with `outputs.forward_output.batch_size` |
| New: `pipeline/phases/explain_detection.py` | reads `forward_output.detection_predictions[i]` (D24) |

**Fix D24 — Single forward pass.** Pipeline phase order ensures `forward_pass` runs before BOTH `evaluate_metrics` and `explain_detection`. Both downstream phases consume the same `ForwardOutput.detection_predictions` so metrics + explanations anchor to identical clean predictions. No duplicate `backend(inputs)` call per sample inside the detection phase.

**Fix D12 — Report-aware single-sample slicing** in `src/raitap/transparency/results.py:render_visualisation_for_scope()`:

```python
if self.original_sample_index is not None:
    # Single-sample detection result already scoped to one sample.
    if sample_index is not None and sample_index != self.original_sample_index:
        return None              # caller (builder.py:714) skips
    # Use stored tensors verbatim — no slicing.
else:
    # Existing classification path — slice by global sample_index.
    if sample_index is not None:
        attributions = attributions[sample_index : sample_index + 1]
        inputs = inputs[sample_index : sample_index + 1]
        ...
```

Same skip-on-mismatch logic mirrored at `builder.py:714` (uses the new method's `None` return to filter out non-matching detection results).

**Fix D13 — Empty-detections behaviour.** Phase logs `raitap_log.warn(...)` with sample_id + threshold + class breakdown so users can adjust. No result emitted. Compatible with downstream code that iterates `RunOutputs.explanations` — list just has K_i < N entries.

### Metrics plumbing (D17′ + D18′)

**Fix D17′ — Forward-pass produces typed `ForwardOutput`** in `src/raitap/pipeline/phases/forward_pass.py`. Per D23 the return type is the new dataclass:

```python
def forward_pass(config, data, backend) -> ForwardOutput:
    raw_outputs = ...   # backend(batch) per chunk

    if backend.task_kind is TaskKind.detection:
        # Detection: per-chunk list[dict[str, Tensor]]. Concatenate into a single
        # flat list of length N (one dict per sample).
        detection_predictions = _concatenate_detection_outputs(raw_outputs)
        return ForwardOutput(
            task_kind=TaskKind.detection,
            batch_size=len(detection_predictions),
            detection_predictions=detection_predictions,
        )

    # Classification path — existing behaviour, packed in the typed dataclass.
    predictions_tensor = extract_primary_tensor(raw_outputs, ...)
    return ForwardOutput(
        task_kind=TaskKind.classification,
        batch_size=int(predictions_tensor.shape[0]),
        predictions_tensor=predictions_tensor,
    )
```

**Fix D18′ — Detection metrics dispatch** in `src/raitap/pipeline/phases/evaluate_metrics.py`:

```python
def evaluate_metrics(config, forward_output: ForwardOutput, labels):
    if not metrics_run_enabled(config):
        return None

    if forward_output.task_kind is TaskKind.detection:
        # Detection: predictions are list[dict]; targets are list[dict] from D22 loader.
        return Metrics(
            config,
            forward_output.detection_predictions,
            labels,
        )

    # Classification path — read predictions_tensor instead of forward_output directly.
    predictions_tensor = forward_output.predictions_tensor
    assert predictions_tensor is not None    # invariant from ForwardOutput.__post_init__
    if (
        getattr(config.metrics, "num_classes", None) is None
        and predictions_tensor.ndim == 2
        and predictions_tensor.shape[1] >= 2
    ):
        config.metrics.num_classes = int(predictions_tensor.shape[1])
    preds, _ = metrics_prediction_pair(predictions_tensor)
    targs = resolve_metric_targets(preds, labels)
    return Metrics(config, preds, targs)
```

Detection targets are produced by `Data._load_detection_labels()` (D22), so `labels` here is already `list[dict[str, Tensor]]` shape-matched to torchmetrics' `MeanAveragePrecision.update()`.

**D20 — Docs hint.** Add one-liner to `docs/modules/metrics/index.md`:

```yaml
# Detection — wraps torchmetrics.detection.MeanAveragePrecision
metrics:
  _target_: DetectionMetrics
```

Plus new entry in `docs/modules/transparency/visualisers.md` describing `DetectionImageVisualiser`.

---

## Data flow (revised)

```
inputs (N, C, H, W)
        │
        ▼
forward_pass(config, data, backend) → ForwardOutput
        │
        ├─ task_kind == classification ─→ ForwardOutput(predictions_tensor=(N, C))
        │                                  ↓
        │                                  evaluate_metrics + prediction_summaries + assess_transparency
        │                                  (existing paths read .predictions_tensor / .batch_size)
        │
        └─ task_kind == detection      ─→ ForwardOutput(detection_predictions=[dict×N])
                                           ↓
                                           ├─ evaluate_metrics:
                                           │     DetectionMetrics.update(
                                           │         forward_output.detection_predictions,
                                           │         labels,                      # list[dict] from D22 loader
                                           │     )
                                           │     → mAP scalars in MetricsEvaluation
                                           │     (reporting deferred — D19)
                                           │
                                           └─ explain_detection phase (D24: no second forward pass):
                                                 for each sample i:
                                                   predictions_i = forward_output.detection_predictions[i]
                                                   mask = predictions_i["scores"] >= score_threshold
                                                   raw_candidates = nonzero(mask).flatten()      # A1 fix
                                                   order = predictions_i["scores"][raw_candidates].argsort(desc)
                                                   top_k_raw_indices = raw_candidates[order[:max_boxes]]
                                                   if top_k_raw_indices.numel() == 0:
                                                     log.warn(sample_id, threshold, max_score); continue
                                                   for j, raw_index in enumerate(top_k_raw_indices.tolist()):
                                                     reference_xyxy = predictions_i["boxes"][raw_index]
                                                     reference_label = predictions_i["labels"][raw_index]
                                                     wrapped = ScalarDetectionWrapper(
                                                       model,
                                                       DetectionTarget(
                                                         mode="reference_match",
                                                         reference_xyxy=reference_xyxy,
                                                         reference_label=reference_label,
                                                         iou_threshold=iou_threshold,
                                                       ),
                                                     )
                                                     call_kwargs["target"] = 0    # D21 normalisation
                                                     result = explainer.explain(
                                                       wrapped, inputs[i:i+1],
                                                       backend=backend,
                                                       run_dir=<base>/sample_{i}/box_{raw_index}/,
                                                       **call_kwargs,
                                                     )
                                                     result.detection_box = DetectionBox(
                                                       display_index=j, raw_index=raw_index,
                                                       xyxy=tuple(reference_xyxy.tolist()),
                                                       score=float(predictions_i["scores"][raw_index]),
                                                       label_index=int(reference_label),
                                                       label_name=None,
                                                     )
                                                     result.original_sample_index = i
                                                     yield result

report (existing builder.py) iterates results:
    render_visualisation_for_scope(visualiser_index, sample_index=summary.sample_index)
    detection result matches sample_index? render. else return None (skip — D12).
    visualiser receives detection_box via context → draws bbox + heatmap + caption.
    K results sharing original_sample_index = i appear consecutively in output (R1).
```

---

## New types reference

| Symbol | Path | Purpose |
|---|---|---|
| `DetectionBox` | `transparency/contracts.py` | Persisted per-box metadata (display/raw index, xyxy, score, label) |
| `VisualisationContext.detection_box` | `transparency/contracts.py` | Render-time channel for the visualiser |
| `ExplanationResult.detection_box` | `transparency/results.py` | Persistence channel; default `None` |
| `ExplanationResult.original_sample_index` | `transparency/results.py` | Maps single-sample detection result to its global sample index |
| `DetectionTarget.mode == "reference_match"` | `models/task_wrappers.py` | IoU+label anchored target (faithful) |
| `DetectionTarget(reference_xyxy, reference_label, iou_threshold)` | `models/task_wrappers.py` | Reference-mode constructor args |
| `_candidate_output_spaces(task_kind=…)` | `transparency/semantics.py` | Task-aware factory pre-flight |
| `explainer_capability(task_kind=…)` | `transparency/semantics.py` | Threaded by factory |
| `ForwardOutput` (typed dataclass) | `pipeline/outputs.py` | Replaces `RunOutputs.forward_output: torch.Tensor`; carries `predictions_tensor`, `detection_predictions`, `batch_size`, `task_kind` (D23) |
| `Data._load_detection_labels()` | `data/data.py` | Returns `list[dict[str, Tensor]]` per-sample boxes+labels (D22) |
| `DetectionImageVisualiser` | `transparency/visualisers/detection_image_visualiser.py` | Renders one fig per box |

---

## Error handling

- **Zero detections after filter**: skip + warn (D13). Log line: `f"sample_id={sample_id}: 0 boxes passed score_threshold={threshold}; max_score={…}; emitting no detection explanations"`. No result yielded.
- **`max_boxes < 1` in config**: validate at config-parse time (pyright type + runtime guard in detection phase).
- **`iou_threshold` outside [0, 1]**: same guard, fail fast at phase entry.
- **Non-detection task wired with `DetectionImageVisualiser`**: pre-flight `validate_explanation` raises via `supported_output_spaces` check (existing `BaseVisualiser` machinery — works once D8 is in).
- **Detection dataset without targets**: `evaluate_metrics` skips with same warning current classification path uses.
- **`reference_match` returns 0 for all perturbations**: explainer computes a flat attribution. Result still emitted (faithful — "the model doesn't see the reference box anywhere under perturbation"). Visualiser caption notes "score=0 under perturbation" if all-zero detected.

---

## Testing strategy

### Unit

- `DetectionTarget(mode="reference_match", …)` — IoU matching with synthetic predictions; matching label, non-matching label, zero-IoU, IoU just below threshold, multiple matches (returns highest-IoU score).
- `DetectionBox` dataclass — round-trip + frozen.
- `ExplanationResult.original_sample_index` — `render_visualisation_for_scope` returns `None` on sample mismatch, returns the stored tensors unchanged on match.
- `_candidate_output_spaces(task_kind=TaskKind.detection)` returns `{DETECTION_BOXES}`.
- `explainer_capability(task_kind=detection)` propagates correctly.
- `DetectionImageVisualiser.visualise()` — synthetic attribution + DetectionBox → figure with rectangle at expected coords, title contains label + score.
- Forward-pass detection branch returns `list[dict]` with correct length and shapes.

### Integration

- End-to-end pipeline: `_FakeDetector` returning K=3 boxes for 2 samples → 6 `ExplanationResult` objects, each with `detection_box` populated, `original_sample_index` matching sample, `run_dir` matching `transparency/{name}/sample_{i}/box_{raw_idx}/`.
- Factory pre-flight: register a `DetectionImageVisualiser` against an explainer with a detection-task backend → passes; same visualiser with classification backend → raises `ValueError` via `BaseVisualiser.validate_explanation()` (existing machinery — the candidate output spaces won't include `DETECTION_BOXES` for a classification task once D8 is in).
- Metrics: `DetectionMetrics` integrated through `evaluate_metrics` with synthetic predictions+targets → returns `MetricsEvaluation` with `mAP` scalar.

### Regression

- Classification end-to-end (existing tests): all current `test_run_main`, `test_e2e_transparency_matrix`, `test_e2e_robustness_matrix` tests still pass — no path touched for `task_kind != detection`.

---

## Files touched (estimate)

| File | Op | Lines (rough) |
|---|---|---|
| `src/raitap/transparency/contracts.py` | modify (add `DetectionBox`, `VisualisationContext.detection_box`) | +25 |
| `src/raitap/transparency/results.py` | modify (`detection_box`, `original_sample_index` fields, render slicing) | +30 |
| `src/raitap/transparency/semantics.py` | modify (`task_kind` kwarg on `_candidate_output_spaces` + `explainer_capability`) | +15 |
| `src/raitap/transparency/factory.py` | modify (reorder backend resolution + thread `task_kind` through pre-flight + plumb detection phase dispatch) — A2 | +25 |
| `src/raitap/transparency/explainers/base_explainer.py` | modify (thread `task_kind` to `infer_output_space`) | +3 |
| `src/raitap/models/task_wrappers.py` | modify (`reference_match` mode in `DetectionTarget`) | +50 |
| `src/raitap/pipeline/phases/explain_detection.py` | new (reads `ForwardOutput.detection_predictions`, no second forward pass) | +140 |
| `src/raitap/pipeline/phases/forward_pass.py` | modify (returns `ForwardOutput`; detection branch packs `list[dict]`) | +30 |
| `src/raitap/pipeline/phases/evaluate_metrics.py` | modify (consume `ForwardOutput`; detection dispatch) | +20 |
| `src/raitap/pipeline/phases/prediction_summaries.py` | modify (guard on `task_kind == classification`) | +5 |
| `src/raitap/pipeline/phases/assess_robustness.py` | modify (guard on `task_kind == classification`) | +5 |
| `src/raitap/pipeline/outputs.py` | modify (new `ForwardOutput` dataclass; `RunOutputs.forward_output` retyped) — D23 | +30 |
| `src/raitap/reporting/builder.py` | modify (`builder.py:1260` reads `.batch_size` not `.shape[0]`) | +3 |
| `src/raitap/data/data.py` | modify (new `_load_detection_labels()` codepath; `data.labels.kind: detection` discriminator) — D22 | +60 |
| `src/raitap/transparency/visualisers/detection_image_visualiser.py` | new | +180 |
| `src/raitap/transparency/visualisers/__init__.py` | modify (re-export) | +1 |
| Tests | new + modify (incl. ForwardOutput dataclass, detection label loader, factory reorder, target normalisation, reference_match) | +800 |
| `docs/modules/transparency/visualisers.md` | modify (add entry) | +20 |
| `docs/modules/transparency/configuration.md` | modify (detection knobs + `auto_pred` rejection note) — D21 | +15 |
| `docs/modules/metrics/index.md` | modify (one-liner) | +5 |

Total: ~1460 lines added, no destructive deletions. Six existing callsites migrate to the typed `ForwardOutput` API.

---

## Out of scope (explicit deferrals)

- **D14** COCO label-name lookup. `label_name=None` stays; future dataset config can populate.
- **D15** Sample-level grouped report section (overview tile). Separate plan after R1 lands in users' hands.
- **D16** DETR / YOLO / non-torchvision detectors. Once Phase 4/5 lands, generalising `_is_torchvision_detection_model` becomes the natural follow-up.
- **D19** Detection metric reporting (mAP table, class-AP table in PDF / HTML). Requires its own design pass on table rendering. Metrics still surface via MLflow tracking + `MetricsEvaluation` programmatic-API return value.
- **Phase 4 (adversarial loss for detection)** — separate plan after Phase 3.
- **Phase 5 (fasterrcnn config + E2E + detection dataset + ONNX export)** — separate plan after Phase 4.

---

## Effort estimate

| Layer | Days |
|---|---|
| `DetectionTarget.reference_match` (IoU + label anchored) + `DetectionBox` + result/context field additions | 2 |
| Task-aware `_candidate_output_spaces` + factory **reorder** (A2) + pre-flight wiring | 1.5 |
| `explain()` threading `backend.task_kind` + DETECTION semantics activation | 0.5 |
| `ForwardOutput` typed dataclass + 6 callsite migrations + tests (D23) | 1.5 |
| `Data._load_detection_labels()` codepath + dataset config discriminator + tests (D22) | 2 |
| Detection pipeline phase (reads `ForwardOutput.detection_predictions`, A1 raw-index filter, K-loop, per-box run_dir, reference construction, target normalisation D21) + tests | 4 |
| `DetectionImageVisualiser` + tests | 2 |
| `ExplanationResult.original_sample_index` + report-aware slicing + tests | 2 |
| Docs (visualiser entry + metrics one-liner + configuration page detection knobs) | 1 |

Total: ~16.5 days ≈ **3 weeks single engineer** — matches the user's revised estimate after the second design-review pass.

---

## Implementation gate checklist (before writing the plan)

- [x] All 7 first-pass review blockers addressed in the design (D6 / D7 / D8 / D9 / D10 / D11 / D12 / D13)
- [x] All 5 second-pass review amendments addressed (A1 raw-index pseudocode, A2 factory reorder, D21 target normalisation, D22 detection label loader, D23 typed `ForwardOutput`, D24 single forward pass)
- [x] All 24 decisions captured in the lock list with provenance
- [x] No placeholder TODOs in any section
- [x] Architecture sections internally consistent (cross-reference of D6 / D8 / D9 / D23 / D24 plumbing matches the data-flow diagram)
- [x] Out-of-scope items explicitly named
- [x] Effort estimate reflects the revised scope (~3 weeks)

When approved: invoke `superpowers:writing-plans` to produce
`docs/superpowers/plans/2026-05-17-detection-image-visualiser.md`.
