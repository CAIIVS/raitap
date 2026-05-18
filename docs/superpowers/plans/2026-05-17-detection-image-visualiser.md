# Detection Image Visualiser + Per-Box Explanation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship one faithful `ExplanationResult` per detected box for any backend whose `task_kind == detection`, rendered by a new `DetectionImageVisualiser`, with minimal metrics plumbing so `DetectionMetrics` runs end-to-end without crashing the pipeline (detection metric reporting in PDF/HTML deferred per D19).

**Architecture:** Three reusable layers from the Phase 1+2 foundation (`TaskKind`, `supported_tasks`, `DetectionTarget`, `ScalarDetectionWrapper`, `DETECTION_BOXES` output space). This plan adds: an IoU-anchored `reference_match` mode to `DetectionTarget`; a typed `ForwardOutput` dataclass (D23) replacing `RunOutputs.forward_output: torch.Tensor`; a detection-aware `Data._load_detection_labels()` (D22); a new pipeline phase `explain_detection.py` that reads pre-computed boxes (D24), filters via `score_threshold` + `max_boxes` using a correct raw-index pattern (A1), and loops the explainer K times per sample with per-box `run_dir` paths (D10); `DetectionImageVisualiser` rendering one figure per box; report-aware `original_sample_index` slicing so K-result-per-sample fan-out plays with the existing `builder.py:1260` rendering loop (D12).

**Tech Stack:** Python 3.13, PyTorch, torchvision detection models, torchmetrics `MeanAveragePrecision`, Hydra / hydra-zen / OmegaConf, pytest, pyright, ruff, `uv run`.

**Spec source:** `docs/superpowers/specs/2026-05-17-detection-image-visualiser-design.md` (24 decisions D1–D24).

---

## File Structure

**New:**
- `src/raitap/pipeline/phases/explain_detection.py` — new pipeline phase, owns the K-loop.
- `src/raitap/transparency/visualisers/detection_image_visualiser.py` — new visualiser.
- `src/raitap/transparency/tests/test_detection_image_visualiser.py` — visualiser unit tests.
- `src/raitap/pipeline/phases/tests/test_explain_detection.py` — phase unit tests.
- `src/raitap/pipeline/tests/test_forward_output.py` — `ForwardOutput` dataclass tests.
- `src/raitap/data/tests/test_detection_labels.py` — detection label loader tests.

**Modified:**
- `src/raitap/models/task_wrappers.py` — `DetectionTarget.reference_match` mode.
- `src/raitap/transparency/contracts.py` — `DetectionBox`, `VisualisationContext.detection_box`.
- `src/raitap/transparency/results.py` — `detection_box` field, `original_sample_index` field, `render_visualisation_for_scope` slicing.
- `src/raitap/transparency/semantics.py` — `task_kind` kwarg on `_candidate_output_spaces` + `explainer_capability`.
- `src/raitap/transparency/factory.py` — backend resolution reordered before semantic compat; pass `task_kind` through.
- `src/raitap/transparency/explainers/base_explainer.py` — thread `backend.task_kind` into `infer_output_space`.
- `src/raitap/pipeline/outputs.py` — new `ForwardOutput` dataclass; `RunOutputs.forward_output` retyped.
- `src/raitap/pipeline/phases/forward_pass.py` — returns `ForwardOutput`; detection branch.
- `src/raitap/pipeline/phases/prediction_summaries.py` — guard on `task_kind == classification`.
- `src/raitap/pipeline/phases/evaluate_metrics.py` — consume `ForwardOutput`; detection dispatch.
- `src/raitap/pipeline/phases/assess_robustness.py` — guard on `task_kind == classification`.
- `src/raitap/pipeline/phases/assess_transparency.py` — delegate to `explain_detection` when detection.
- `src/raitap/reporting/builder.py` — read `.batch_size` not `.shape[0]` at line 1260.
- `src/raitap/data/data.py` — new `_load_detection_labels`; discriminator `data.labels.kind`.
- `src/raitap/transparency/visualisers/__init__.py` — re-export `DetectionImageVisualiser`.
- `docs/modules/transparency/visualisers.md` — new entry.
- `docs/modules/transparency/configuration.md` — detection knobs section + `auto_pred` rejection note.
- `docs/modules/metrics/index.md` — one-liner detection example.

**Tests modified:**
- `src/raitap/transparency/tests/test_semantics.py` — `task_kind` plumbing.
- `src/raitap/transparency/tests/test_factory.py` — factory reorder + pre-flight detection acceptance.
- `src/raitap/transparency/explainers/tests/test_base_explainer.py` — `task_kind` threading.
- `src/raitap/models/tests/test_task_wrappers.py` — `reference_match` mode.
- `src/raitap/pipeline/tests/test_forward_pass.py` (existing or new) — detection branch.
- `src/raitap/reporting/tests/test_builder.py` — `.batch_size` usage.

---

## Conventions

- All tests via `uv run pytest <path> -v`.
- Format via `uv run ruff format .` after edits batch.
- Type-check via `uv run pyright <paths>`.
- Commits: Conventional Commits (`feat:`, `test:`, `refactor:`, `fix:`). No `Co-Authored-By:` trailer.
- Each Task ends with one commit covering the task's tests + impl. Mark complete only after the commit lands AND `uv run pytest src/raitap` passes for the touched module subtree.
- For tasks that touch multiple files, single commit per task is fine.

---

### Task 1: `DetectionTarget.reference_match` — IoU + label anchored target (D6)

**Files:**
- Modify: `src/raitap/models/task_wrappers.py`
- Modify: `src/raitap/models/tests/test_task_wrappers.py` — append

- [ ] **Step 1: Write the failing tests**

Append to `src/raitap/models/tests/test_task_wrappers.py` (use `-> None` annotations, ruff requires them):

```python
def test_reference_match_returns_score_of_best_iou_predicted_box() -> None:
    target = DetectionTarget(
        mode="reference_match",
        reference_xyxy=(0.0, 0.0, 10.0, 10.0),
        reference_label=1,
        iou_threshold=0.5,
    )
    out = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 9.0, 9.0], [50.0, 50.0, 60.0, 60.0]]),
            "scores": torch.tensor([0.8, 0.4]),
            "labels": torch.tensor([1, 1]),
        },
    ]
    value = target(out)
    assert torch.allclose(value, torch.tensor(0.8))


def test_reference_match_zero_when_no_iou_match() -> None:
    target = DetectionTarget(
        mode="reference_match",
        reference_xyxy=(0.0, 0.0, 10.0, 10.0),
        reference_label=1,
        iou_threshold=0.5,
    )
    out = [
        {
            "boxes": torch.tensor([[50.0, 50.0, 60.0, 60.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1]),
        },
    ]
    value = target(out)
    assert torch.allclose(value, torch.tensor(0.0))


def test_reference_match_zero_when_label_mismatch() -> None:
    target = DetectionTarget(
        mode="reference_match",
        reference_xyxy=(0.0, 0.0, 10.0, 10.0),
        reference_label=1,
        iou_threshold=0.5,
    )
    out = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 9.0, 9.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([2]),
        },
    ]
    value = target(out)
    assert torch.allclose(value, torch.tensor(0.0))


def test_reference_match_picks_highest_iou_when_multiple_matches() -> None:
    target = DetectionTarget(
        mode="reference_match",
        reference_xyxy=(0.0, 0.0, 10.0, 10.0),
        reference_label=1,
        iou_threshold=0.3,
    )
    out = [
        {
            "boxes": torch.tensor(
                [[0.0, 0.0, 5.0, 5.0], [0.0, 0.0, 9.0, 9.0]],
            ),
            "scores": torch.tensor([0.95, 0.6]),
            "labels": torch.tensor([1, 1]),
        },
    ]
    value = target(out)
    # The second box has higher IoU with the reference (~0.81 vs ~0.25), so
    # 0.6 wins despite the lower score.
    assert torch.allclose(value, torch.tensor(0.6))


def test_reference_match_requires_reference_xyxy_and_label() -> None:
    with pytest.raises(ValueError, match="reference_xyxy"):
        DetectionTarget(mode="reference_match")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="reference_label"):
        DetectionTarget(mode="reference_match", reference_xyxy=(0.0, 0.0, 1.0, 1.0))  # type: ignore[call-arg]


def test_reference_match_iou_threshold_default_is_half() -> None:
    target = DetectionTarget(
        mode="reference_match",
        reference_xyxy=(0.0, 0.0, 10.0, 10.0),
        reference_label=1,
    )
    assert target.iou_threshold == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/raitap/models/tests/test_task_wrappers.py -v -k reference_match`
Expected: FAIL — `DetectionTarget` does not accept `reference_match` mode or the new kwargs.

- [ ] **Step 3: Extend `DetectionTarget`**

Edit `src/raitap/models/task_wrappers.py`. Replace the `DetectionTargetMode` alias and the `_VALID_MODES` frozenset, then update `DetectionTarget.__init__` and `__call__`. After this step the full class should look like:

```python
DetectionTargetMode = Literal["class_score", "objectness", "bbox_l2", "reference_match"]
_VALID_MODES: frozenset[str] = frozenset(
    {"class_score", "objectness", "bbox_l2", "reference_match"}
)


def _box_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between a single box ``box_a`` (shape (4,)) and an array of
    boxes ``box_b`` (shape (N, 4)). Inputs in xyxy. Returns shape (N,)."""
    x1 = torch.maximum(box_a[0], box_b[:, 0])
    y1 = torch.maximum(box_a[1], box_b[:, 1])
    x2 = torch.minimum(box_a[2], box_b[:, 2])
    y2 = torch.minimum(box_a[3], box_b[:, 3])
    inter_w = (x2 - x1).clamp(min=0.0)
    inter_h = (y2 - y1).clamp(min=0.0)
    inter = inter_w * inter_h
    area_a = (box_a[2] - box_a[0]).clamp(min=0.0) * (box_a[3] - box_a[1]).clamp(min=0.0)
    area_b = (box_b[:, 2] - box_b[:, 0]).clamp(min=0.0) * (box_b[:, 3] - box_b[:, 1]).clamp(min=0.0)
    union = area_a + area_b - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


class DetectionTarget:
    """Reduce a torchvision-style detection output to a scalar tensor.

    Parameters
    ----------
    mode:
        ``"class_score"`` — score at output-list index ``box_idx`` summed across
        the batch. Indexes the detector output AS-EMITTED — not a stable
        semantic reference; useful for raw-output debugging.
        ``"objectness"`` — sum of all box scores across the batch.
        ``"bbox_l2"`` — squared L2 norm of the first sample's ``box_idx``-th box.
        ``"reference_match"`` — score of the predicted box whose IoU with
        ``reference_xyxy`` is highest, restricted to predictions whose label
        equals ``reference_label`` and whose IoU is at least ``iou_threshold``.
        Returns ``0.0`` if no match passes the threshold. This is the faithful
        per-box mode used by the detection explain phase: the target stays
        anchored to a specific reference box across perturbations rather than
        drifting with output-list reordering.
    box_idx:
        Required for ``class_score`` / ``objectness`` / ``bbox_l2``. Ignored
        for ``reference_match``.
    reference_xyxy:
        Required for ``reference_match``. xyxy coordinates of the reference box.
    reference_label:
        Required for ``reference_match``. Torchvision class id.
    iou_threshold:
        Used by ``reference_match`` only. Defaults to ``0.5``.
    """

    def __init__(
        self,
        mode: DetectionTargetMode,
        *,
        box_idx: int = 0,
        reference_xyxy: tuple[float, float, float, float] | None = None,
        reference_label: int | None = None,
        iou_threshold: float = 0.5,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"DetectionTarget mode must be one of {sorted(_VALID_MODES)}; got {mode!r}."
            )
        if mode == "reference_match":
            if reference_xyxy is None:
                raise ValueError(
                    "DetectionTarget(mode='reference_match') requires reference_xyxy."
                )
            if reference_label is None:
                raise ValueError(
                    "DetectionTarget(mode='reference_match') requires reference_label."
                )
            if not 0.0 <= iou_threshold <= 1.0:
                raise ValueError(
                    f"iou_threshold must lie in [0, 1]; got {iou_threshold!r}."
                )
        self.mode = mode
        self.box_idx = int(box_idx)
        self.reference_xyxy = (
            tuple(float(x) for x in reference_xyxy) if reference_xyxy is not None else None
        )
        self.reference_label = int(reference_label) if reference_label is not None else None
        self.iou_threshold = float(iou_threshold)

    def __call__(self, model_out: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        if not isinstance(model_out, list) or (
            model_out and not isinstance(model_out[0], dict)
        ):
            raise TypeError(
                "DetectionTarget expects list[dict[str, Tensor]] from a torchvision "
                f"detection model; got {type(model_out).__name__}."
            )
        if not model_out:
            return torch.tensor(0.0)

        if self.mode == "objectness":
            per_sample_sums: list[torch.Tensor] = []
            for item in model_out:
                scores = item.get("scores")
                if scores is None or scores.numel() == 0:
                    device = scores.device if scores is not None else None
                    per_sample_sums.append(torch.tensor(0.0, device=device))
                else:
                    per_sample_sums.append(scores.sum())
            return torch.stack(per_sample_sums).sum()

        if self.mode == "class_score":
            per_sample: list[torch.Tensor] = []
            for item in model_out:
                scores = item.get("scores")
                if scores is None or scores.numel() <= self.box_idx:
                    device = scores.device if scores is not None else None
                    per_sample.append(torch.tensor(0.0, device=device))
                else:
                    per_sample.append(scores[self.box_idx])
            return torch.stack(per_sample).sum()

        if self.mode == "bbox_l2":
            boxes = model_out[0].get("boxes")
            if boxes is None or boxes.numel() == 0 or boxes.shape[0] <= self.box_idx:
                return torch.tensor(0.0)
            return (boxes[self.box_idx] ** 2).sum()

        # mode == "reference_match"
        assert self.reference_xyxy is not None
        assert self.reference_label is not None
        per_sample_scores: list[torch.Tensor] = []
        for item in model_out:
            boxes = item.get("boxes")
            scores = item.get("scores")
            labels = item.get("labels")
            if boxes is None or scores is None or labels is None or boxes.shape[0] == 0:
                device = (boxes or scores or labels).device if any(
                    t is not None for t in (boxes, scores, labels)
                ) else None
                per_sample_scores.append(torch.tensor(0.0, device=device))
                continue
            reference = torch.tensor(self.reference_xyxy, device=boxes.device, dtype=boxes.dtype)
            ious = _box_iou(reference, boxes)
            label_mask = labels == self.reference_label
            iou_mask = ious >= self.iou_threshold
            combined = label_mask & iou_mask
            if not combined.any():
                per_sample_scores.append(torch.tensor(0.0, device=boxes.device))
                continue
            masked_ious = torch.where(combined, ious, torch.full_like(ious, -1.0))
            best_idx = int(torch.argmax(masked_ious).item())
            per_sample_scores.append(scores[best_idx])
        return torch.stack(per_sample_scores).sum()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/raitap/models/tests/test_task_wrappers.py -v`
Expected: all (existing 12 + 6 new) pass.

- [ ] **Step 5: Pyright on the file**

Run: `uv run pyright src/raitap/models/task_wrappers.py src/raitap/models/tests/test_task_wrappers.py`
Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add src/raitap/models/task_wrappers.py src/raitap/models/tests/test_task_wrappers.py
git commit -m "feat(models): add DetectionTarget reference_match mode

IoU + label anchored target so per-box attributions stay tied to a
specific reference box across perturbations, instead of drifting with
output-list reordering. Issue #146 Phase 3 (D6)."
```

---

### Task 2: `DetectionBox` dataclass + `VisualisationContext.detection_box`

**Files:**
- Modify: `src/raitap/transparency/contracts.py`
- Modify: `src/raitap/transparency/tests/test_contracts.py` — append

- [ ] **Step 1: Write the failing tests**

Append to `src/raitap/transparency/tests/test_contracts.py`:

```python
def test_detection_box_round_trip() -> None:
    from raitap.transparency.contracts import DetectionBox

    box = DetectionBox(
        display_index=0,
        raw_index=7,
        xyxy=(1.0, 2.0, 3.0, 4.0),
        score=0.93,
        label_index=5,
        label_name="car",
    )
    assert box.display_index == 0
    assert box.raw_index == 7
    assert box.xyxy == (1.0, 2.0, 3.0, 4.0)
    assert box.score == 0.93
    assert box.label_index == 5
    assert box.label_name == "car"


def test_detection_box_label_name_defaults_to_none() -> None:
    from raitap.transparency.contracts import DetectionBox

    box = DetectionBox(
        display_index=0, raw_index=0, xyxy=(0.0, 0.0, 1.0, 1.0), score=0.5, label_index=1
    )
    assert box.label_name is None


def test_detection_box_is_frozen() -> None:
    from raitap.transparency.contracts import DetectionBox

    box = DetectionBox(
        display_index=0, raw_index=0, xyxy=(0.0, 0.0, 1.0, 1.0), score=0.5, label_index=1
    )
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        box.score = 0.9  # type: ignore[misc]


def test_visualisation_context_detection_box_defaults_to_none() -> None:
    from raitap.transparency.contracts import VisualisationContext

    ctx = VisualisationContext(algorithm="x", sample_names=None, show_sample_names=False)
    assert ctx.detection_box is None
```

Make sure `import dataclasses` is at the top of the test file. If missing, add it.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/raitap/transparency/tests/test_contracts.py -v -k "detection_box or visualisation_context_detection"`
Expected: FAIL — `DetectionBox` does not exist; `VisualisationContext` has no `detection_box` field.

- [ ] **Step 3: Add `DetectionBox` + extend `VisualisationContext`**

Edit `src/raitap/transparency/contracts.py`. Add this dataclass immediately after `class ExplanationOutputSpace(StrEnum):` (and its members) — search for `class VisualSummarySpec` and place `DetectionBox` immediately before it:

```python
@dataclass(frozen=True)
class DetectionBox:
    """Per-box metadata persisted with a detection explanation result.

    ``display_index`` is the rank in the filtered set (0..K-1) and is the
    user-facing ordinal. ``raw_index`` is the index in the clean forward
    pass's output list — useful for provenance and for re-anchoring the
    explanation to the original detection. ``xyxy`` is in input pixel space.
    """

    display_index: int
    raw_index: int
    xyxy: tuple[float, float, float, float]
    score: float
    label_index: int
    label_name: str | None = None
```

Then extend `VisualisationContext` (currently `algorithm`, `sample_names`, `show_sample_names`):

```python
@dataclass(frozen=True)
class VisualisationContext:
    """
    Standard RAITAP metadata provided to visualisers during the assessment pipeline.
    """

    algorithm: str
    sample_names: list[str] | None
    show_sample_names: bool
    detection_box: DetectionBox | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/raitap/transparency/tests/test_contracts.py -v`
Expected: all pass.

- [ ] **Step 5: Pyright on the file**

Run: `uv run pyright src/raitap/transparency/contracts.py src/raitap/transparency/tests/test_contracts.py`
Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add src/raitap/transparency/contracts.py src/raitap/transparency/tests/test_contracts.py
git commit -m "feat(transparency): add DetectionBox + VisualisationContext.detection_box

Typed per-box metadata channel. display_index = rank in filtered set
(0..K-1); raw_index = index in clean forward output. VisualisationContext
gains an optional detection_box for render-time propagation. Issue #146
Phase 3 (D7)."
```

---

### Task 3: `ExplanationResult.detection_box` + `original_sample_index` + render slicing (D12)

**Files:**
- Modify: `src/raitap/transparency/results.py`
- Modify: `src/raitap/transparency/tests/test_results.py` — append

- [ ] **Step 1: Write the failing tests**

Append to `src/raitap/transparency/tests/test_results.py`:

```python
def test_explanation_result_detection_box_defaults_to_none(
    tmp_path,
    explanation_factory,    # existing fixture in test_results.py producing a minimal ExplanationResult
) -> None:
    result = explanation_factory(tmp_path)
    assert result.detection_box is None
    assert result.original_sample_index is None


def test_explanation_result_render_visualisation_for_scope_returns_none_on_sample_mismatch(
    tmp_path,
    explanation_factory,
) -> None:
    from raitap.transparency.contracts import DetectionBox

    result = explanation_factory(tmp_path)
    result.original_sample_index = 0
    result.detection_box = DetectionBox(
        display_index=0,
        raw_index=3,
        xyxy=(0.0, 0.0, 1.0, 1.0),
        score=0.9,
        label_index=1,
    )
    # Result is scoped to sample 0 — requesting sample 3 must return None.
    out = result.render_visualisation_for_scope(
        visualiser_index=0, scope="local", sample_index=3
    )
    assert out is None


def test_explanation_result_propagates_detection_box_to_context(
    tmp_path,
    explanation_factory,
    capture_visualisation_context,    # new fixture, see below
) -> None:
    from raitap.transparency.contracts import DetectionBox

    box = DetectionBox(
        display_index=2, raw_index=11, xyxy=(0.0, 0.0, 2.0, 2.0), score=0.7, label_index=4
    )
    result = explanation_factory(tmp_path)
    result.original_sample_index = 0
    result.detection_box = box

    result.render_visualisation_for_scope(
        visualiser_index=0, scope="local", sample_index=0
    )

    captured = capture_visualisation_context.captured
    assert captured is not None
    assert captured.detection_box == box
```

If `explanation_factory` and `capture_visualisation_context` fixtures don't exist, add them at the top of `test_results.py` (near other fixtures):

```python
import pytest
from raitap.transparency.contracts import (
    ExplanationOutputSpace,
    ExplanationPayloadKind,
    ExplanationScope,
    InputKind,
    InputSpec,
    OutputSpaceSpec,
    ScopeDefinitionStep,
    TensorLayout,
    VisualisationContext,
)
from raitap.transparency.results import (
    ConfiguredVisualiser,
    ExplanationResult,
)
from raitap.transparency.contracts import ExplanationSemantics
import torch


class _RecordingVisualiser:
    """Test double for BaseVisualiser; captures the VisualisationContext passed in."""

    captured: VisualisationContext | None = None

    def visualise(
        self,
        attributions,
        inputs=None,
        *,
        context=None,
        **kwargs,
    ):
        type(self).captured = context
        from matplotlib.figure import Figure
        return Figure()

    def validate_explanation(self, explanation, attributions, inputs) -> None:
        return None

    def save(self, attributions, output_path, inputs=None, *, context=None, **kwargs) -> None:
        return None


@pytest.fixture
def capture_visualisation_context():
    _RecordingVisualiser.captured = None
    return _RecordingVisualiser


@pytest.fixture
def explanation_factory(capture_visualisation_context):
    def _make(tmp_path):
        attributions = torch.zeros(1, 3, 4, 4)
        inputs = torch.zeros(1, 3, 4, 4)
        input_spec = InputSpec(
            kind=InputKind.IMAGE,
            shape=(1, 3, 4, 4),
            layout=TensorLayout.BATCH_CHANNEL_HEIGHT_WIDTH,
        )
        semantics = ExplanationSemantics(
            scope=ExplanationScope.LOCAL,
            scope_definition_step=ScopeDefinitionStep.EXPLAINER_OUTPUT,
            payload_kind=ExplanationPayloadKind.ATTRIBUTIONS,
            method_families=frozenset(),
            target=None,
            sample_selection=None,
            input_spec=input_spec,
            output_space=OutputSpaceSpec(
                space=ExplanationOutputSpace.INPUT_FEATURES,
                shape=(1, 3, 4, 4),
                layout=TensorLayout.BATCH_CHANNEL_HEIGHT_WIDTH,
            ),
        )
        return ExplanationResult(
            attributions=attributions,
            inputs=inputs,
            run_dir=tmp_path,
            experiment_name=None,
            explainer_target="x",
            algorithm="alg",
            visualisers=[ConfiguredVisualiser(visualiser=capture_visualisation_context())],
            semantics=semantics,
        )

    return _make
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/raitap/transparency/tests/test_results.py -v -k "detection_box or original_sample_index or render_visualisation_for_scope_returns_none"`
Expected: FAIL — fields don't exist; slicing always proceeds.

- [ ] **Step 3: Extend `ExplanationResult`**

Edit `src/raitap/transparency/results.py`. After the existing dataclass field block (around `semantics: ExplanationSemantics = field(kw_only=True)`) add the two new fields:

```python
@dataclass
class ExplanationResult(Trackable):
    attributions: torch.Tensor
    inputs: torch.Tensor
    run_dir: Path
    experiment_name: str | None
    explainer_target: str
    algorithm: str
    explainer_name: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    call_kwargs: dict[str, Any] = field(default_factory=dict)
    visualiser_targets: list[str] = field(default_factory=list)
    visualisers: list[ConfiguredVisualiser] = field(default_factory=list, repr=False)
    payload_kind: ExplanationPayloadKind = ExplanationPayloadKind.ATTRIBUTIONS
    detection_box: DetectionBox | None = None                # NEW (D7)
    original_sample_index: int | None = None                 # NEW (D12)
    semantics: ExplanationSemantics = field(kw_only=True)
```

`DetectionBox` import at the top of the file: `from raitap.transparency.contracts import DetectionBox` (if not already imported alongside `ExplanationSemantics`).

- [ ] **Step 4: Update `render_visualisation_for_scope` slicing**

Find the existing `if sample_index is not None:` slicing block in `render_visualisation_for_scope` (around `results.py:327` per the spec). Replace with:

```python
        if self.original_sample_index is not None:
            # Single-sample detection result. The result already represents
            # exactly one sample (i.e. attributions shape is (1, ...)); the
            # caller iterates over global sample indices, so skip when the
            # requested sample doesn't match this result's owner.
            if sample_index is not None and sample_index != self.original_sample_index:
                return None
            # Match (or no specific sample requested) → use stored tensors as-is.
        elif sample_index is not None:
            attributions = attributions[sample_index : sample_index + 1]
            inputs = inputs[sample_index : sample_index + 1]
            if sample_names:
                sample_names = sample_names[sample_index : sample_index + 1]
        else:
            limit = _batch_size(attributions) or _batch_size(inputs)
            if limit is not None:
                sample_names = sample_names[:limit]
```

(Preserve the rest of the method exactly as it was — the only change is replacing the `if sample_index is not None:` branch with the elif-pattern above.)

- [ ] **Step 5: Propagate `detection_box` into `VisualisationContext`**

In the same method, find the existing `context = VisualisationContext(algorithm=..., sample_names=..., show_sample_names=...)` line and replace with:

```python
        context = VisualisationContext(
            algorithm=self.algorithm,
            sample_names=sample_names,
            show_sample_names=show_sample_names,
            detection_box=self.detection_box,
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest src/raitap/transparency/tests/test_results.py -v`
Expected: all (existing + 3 new) pass.

- [ ] **Step 7: Run the transparency regression sweep**

Run: `uv run pytest src/raitap/transparency -x -q`
Expected: all pass — the new fields default to `None`, the slicing change preserves the existing classification path bit-for-bit.

- [ ] **Step 8: Pyright**

Run: `uv run pyright src/raitap/transparency/results.py src/raitap/transparency/tests/test_results.py`
Expected: 0 errors.

- [ ] **Step 9: Commit**

```bash
git add src/raitap/transparency/results.py src/raitap/transparency/tests/test_results.py
git commit -m "feat(transparency): ExplanationResult per-box fields + report slicing

Two new optional fields: detection_box (typed bbox+score metadata) and
original_sample_index (which global sample this single-sample result
belongs to). render_visualisation_for_scope skips when sample_index
doesn't match original_sample_index; uses stored tensors verbatim when it
matches. Classification path unchanged. Issue #146 Phase 3 (D12)."
```

---

### Task 4: Task-aware candidate output spaces (D8)

**Files:**
- Modify: `src/raitap/transparency/semantics.py`
- Modify: `src/raitap/transparency/tests/test_semantics.py` — append

- [ ] **Step 1: Write the failing tests**

Append to `src/raitap/transparency/tests/test_semantics.py`:

```python
def test_candidate_output_spaces_returns_detection_boxes_when_task_is_detection() -> None:
    from raitap.transparency.semantics import _candidate_output_spaces
    from raitap.types import TaskKind

    result = _candidate_output_spaces(frozenset(), task_kind=TaskKind.detection)
    assert result == frozenset({ExplanationOutputSpace.DETECTION_BOXES})


def test_candidate_output_spaces_classification_unchanged_when_task_kind_omitted() -> None:
    from raitap.transparency.semantics import _candidate_output_spaces

    result = _candidate_output_spaces(frozenset({MethodFamily.GRADIENT}))
    assert ExplanationOutputSpace.INPUT_FEATURES in result
    assert ExplanationOutputSpace.DETECTION_BOXES not in result


def test_explainer_capability_threads_task_kind_for_detection(monkeypatch) -> None:
    from raitap.transparency.explainers.captum_explainer import CaptumExplainer
    from raitap.transparency.semantics import explainer_capability
    from raitap.types import TaskKind

    explainer = CaptumExplainer(algorithm="IntegratedGradients")
    capability = explainer_capability(explainer, task_kind=TaskKind.detection)
    assert capability.candidate_output_spaces == frozenset(
        {ExplanationOutputSpace.DETECTION_BOXES}
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/raitap/transparency/tests/test_semantics.py -v -k "candidate_output_spaces or explainer_capability_threads_task_kind"`
Expected: FAIL — `_candidate_output_spaces` doesn't accept `task_kind`; `explainer_capability` doesn't accept `task_kind`.

- [ ] **Step 3: Extend `_candidate_output_spaces` + `explainer_capability`**

Edit `src/raitap/transparency/semantics.py`. Find `_candidate_output_spaces` (around line 315 per the spec recon) and replace:

```python
def _candidate_output_spaces(
    method_families: frozenset[MethodFamily],
    task_kind: TaskKind | None = None,
) -> frozenset[ExplanationOutputSpace]:
    if task_kind is TaskKind.detection:
        return frozenset({ExplanationOutputSpace.DETECTION_BOXES})
    if MethodFamily.CAM in method_families:
        return frozenset(
            {
                ExplanationOutputSpace.IMAGE_SPATIAL_MAP,
                ExplanationOutputSpace.LAYER_ACTIVATION,
            }
        )
    return frozenset(
        {
            ExplanationOutputSpace.INPUT_FEATURES,
            ExplanationOutputSpace.INTERPRETABLE_FEATURES,
            ExplanationOutputSpace.TOKEN_SEQUENCE,
        }
    )
```

Find `explainer_capability` (around line 103) and extend:

```python
def explainer_capability(
    explainer: object, *, task_kind: TaskKind | None = None
) -> ExplainerCapability:
    """Return broad pre-compute semantic capabilities for an explainer."""

    method_families = method_families_for_explainer(explainer)
    return ExplainerCapability(
        scope=explainer_output_scope(explainer),
        scope_definition_step=ScopeDefinitionStep.EXPLAINER_OUTPUT,
        payload_kind=explainer_output_kind(explainer),
        method_families=method_families,
        candidate_output_spaces=_candidate_output_spaces(method_families, task_kind=task_kind),
    )
```

(`TaskKind` is already imported at semantics.py:8 from PR #176.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/raitap/transparency/tests/test_semantics.py -v`
Expected: all pass.

- [ ] **Step 5: Transparency regression sweep**

Run: `uv run pytest src/raitap/transparency -x -q`
Expected: all pass.

- [ ] **Step 6: Pyright**

Run: `uv run pyright src/raitap/transparency/semantics.py src/raitap/transparency/tests/test_semantics.py`
Expected: 0 errors.

- [ ] **Step 7: Commit**

```bash
git add src/raitap/transparency/semantics.py src/raitap/transparency/tests/test_semantics.py
git commit -m "feat(transparency): task-aware candidate output spaces

_candidate_output_spaces and explainer_capability accept an optional
task_kind kwarg. When detection, candidate spaces are
{DETECTION_BOXES} so the factory pre-flight no longer rejects
DetectionImageVisualiser. Classification path unchanged. Issue #146
Phase 3 (D8)."
```

---

### Task 5: Thread `backend.task_kind` through `AttributionOnlyExplainer.explain()` (D9)

**Files:**
- Modify: `src/raitap/transparency/explainers/base_explainer.py`
- Modify: `src/raitap/transparency/explainers/tests/test_base_explainer.py` — append

- [ ] **Step 1: Write the failing test**

Append to `src/raitap/transparency/explainers/tests/test_base_explainer.py`:

```python
def test_attribution_only_explainer_threads_backend_task_kind_into_infer_output_space(
    tmp_path,
) -> None:
    """When the backend reports task_kind=detection, the inferred output space
    must be DETECTION_BOXES (DETECTION branch in infer_output_space)."""
    import torch
    from raitap.models.backend import TorchBackend
    from raitap.transparency.contracts import ExplanationOutputSpace
    from raitap.types import TaskKind

    # Fake explainer wraps the BaseExplainer pipeline to capture semantics
    # without depending on Captum / SHAP libraries.
    from raitap.transparency.explainers.base_explainer import AttributionOnlyExplainer
    from raitap.transparency.explainers.registration import register_transparency_adapter
    from raitap.transparency.contracts import MethodFamily

    class _IdentityAttrModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 1)

    backend = TorchBackend(_IdentityAttrModel(), task_kind=TaskKind.detection)

    @register_transparency_adapter(
        registry_name="_test_detection_explainer",
        algorithm_registry={"Id": frozenset({MethodFamily.GRADIENT})},
        extra="test",
    )
    class _IdExplainer(AttributionOnlyExplainer):
        def __init__(self, algorithm: str = "Id") -> None:
            super().__init__()
            self.algorithm = algorithm

        def compute_attributions(self, model, inputs, **kwargs):
            return torch.zeros_like(inputs)

    explainer = _IdExplainer()
    result = explainer.explain(
        backend.as_model_for_explanation(),
        torch.zeros(1, 3, 4, 4),
        backend=backend,
        run_dir=tmp_path,
        explainer_target="t",
        explainer_name="_IdExplainer",
        raitap_kwargs={
            "input_metadata": {"kind": "image", "layout": "NCHW"},
        },
    )
    assert result.semantics.output_space.space is ExplanationOutputSpace.DETECTION_BOXES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/raitap/transparency/explainers/tests/test_base_explainer.py -v -k threads_backend_task_kind`
Expected: FAIL — result's output_space is `INPUT_FEATURES` because `task_kind` not threaded.

- [ ] **Step 3: Thread `task_kind` in `explain()`**

Edit `src/raitap/transparency/explainers/base_explainer.py:138`. Find:

```python
        output_space = infer_output_space(
            input_spec=input_spec,
            attributions=attributions,
            explainer=self,
            method_families=method_families,
            layer_path=_layer_path_for_explainer(self),
        )
```

Replace with:

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

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/raitap/transparency/explainers/tests/test_base_explainer.py -v -k threads_backend_task_kind`
Expected: pass.

- [ ] **Step 5: Transparency regression sweep**

Run: `uv run pytest src/raitap/transparency -x -q`
Expected: all pass — `getattr(backend, "task_kind", None)` returns `None` when `backend` is unset (legacy direct nn.Module use), preserving the classification default.

- [ ] **Step 6: Commit**

```bash
git add src/raitap/transparency/explainers/base_explainer.py src/raitap/transparency/explainers/tests/test_base_explainer.py
git commit -m "feat(transparency): thread backend.task_kind into infer_output_space

The DETECTION branch added in PR #176 is otherwise dead code — the
explainer pipeline must forward task_kind from the backend when calling
infer_output_space. Detection backends now correctly tag semantics with
DETECTION_BOXES output space. Issue #146 Phase 3 (D9)."
```

---

### Task 6: Factory reorder + `task_kind` plumbing into pre-flight (A2)

**Files:**
- Modify: `src/raitap/transparency/factory.py`
- Modify: `src/raitap/transparency/tests/test_factory.py` — append

- [ ] **Step 1: Write the failing test**

Append to `src/raitap/transparency/tests/test_factory.py`:

```python
def test_factory_preflight_accepts_detection_visualiser_for_detection_backend(
    tmp_path, monkeypatch
) -> None:
    """A DetectionImageVisualiser (supported_output_spaces = {DETECTION_BOXES})
    must pass the factory pre-flight when the backend's task_kind is
    detection — requires the factory to resolve the backend BEFORE the
    semantic compat check, AND to thread task_kind into explainer_capability."""
    import torch
    from torch import nn
    from raitap.models.backend import TorchBackend
    from raitap.transparency.contracts import (
        ConfiguredVisualiser,
        ExplanationOutputSpace,
        ExplanationPayloadKind,
        ExplanationScope,
    )
    from raitap.transparency.factory import check_explainer_visualiser_semantic_compat
    from raitap.transparency.visualisers.base_visualiser import BaseVisualiser
    from raitap.types import TaskKind
    from raitap.transparency.contracts import MethodFamily

    class _DummyVisualiser(BaseVisualiser):
        supported_output_spaces = frozenset({ExplanationOutputSpace.DETECTION_BOXES})
        supported_payload_kinds = frozenset({ExplanationPayloadKind.ATTRIBUTIONS})
        supported_scopes = frozenset({ExplanationScope.LOCAL})
        supported_method_families = frozenset({MethodFamily.GRADIENT})

        def visualise(self, attributions, inputs=None, *, context=None, **kwargs):
            from matplotlib.figure import Figure
            return Figure()

    class _DummyExplainer:
        algorithm = "IntegratedGradients"
        algorithm_registry = {"IntegratedGradients": frozenset({MethodFamily.GRADIENT})}
        output_payload_kind = ExplanationPayloadKind.ATTRIBUTIONS
        output_scope = ExplanationScope.LOCAL

    class _Linear(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 1)

    backend = TorchBackend(_Linear(), task_kind=TaskKind.detection)

    # MUST not raise — backend resolves first, task_kind=detection makes
    # DETECTION_BOXES a candidate, so the visualiser's supported_output_spaces
    # intersects the candidates.
    check_explainer_visualiser_semantic_compat(
        explainer=_DummyExplainer(),
        explainer_target="t",
        visualisers=[ConfiguredVisualiser(visualiser=_DummyVisualiser())],
        task_kind=backend.task_kind,
    )


def test_factory_preflight_rejects_detection_visualiser_for_classification_backend(
    tmp_path,
) -> None:
    import torch
    from torch import nn
    from raitap.models.backend import TorchBackend
    from raitap.transparency.contracts import (
        ConfiguredVisualiser,
        ExplanationOutputSpace,
        ExplanationPayloadKind,
        ExplanationScope,
        MethodFamily,
    )
    from raitap.transparency.factory import check_explainer_visualiser_semantic_compat
    from raitap.transparency.visualisers.base_visualiser import BaseVisualiser
    from raitap.types import TaskKind

    class _DummyVisualiser(BaseVisualiser):
        supported_output_spaces = frozenset({ExplanationOutputSpace.DETECTION_BOXES})
        supported_payload_kinds = frozenset({ExplanationPayloadKind.ATTRIBUTIONS})
        supported_scopes = frozenset({ExplanationScope.LOCAL})
        supported_method_families = frozenset({MethodFamily.GRADIENT})

        def visualise(self, attributions, inputs=None, *, context=None, **kwargs):
            from matplotlib.figure import Figure
            return Figure()

    class _DummyExplainer:
        algorithm = "IntegratedGradients"
        algorithm_registry = {"IntegratedGradients": frozenset({MethodFamily.GRADIENT})}
        output_payload_kind = ExplanationPayloadKind.ATTRIBUTIONS
        output_scope = ExplanationScope.LOCAL

    class _Linear(nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 2)

    backend = TorchBackend(_Linear(), task_kind=TaskKind.classification)

    import pytest as _pytest
    with _pytest.raises(ValueError, match="output space"):
        check_explainer_visualiser_semantic_compat(
            explainer=_DummyExplainer(),
            explainer_target="t",
            visualisers=[ConfiguredVisualiser(visualiser=_DummyVisualiser())],
            task_kind=backend.task_kind,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/raitap/transparency/tests/test_factory.py -v -k preflight`
Expected: FAIL — `check_explainer_visualiser_semantic_compat` doesn't accept `task_kind`.

- [ ] **Step 3: Extend `check_explainer_visualiser_semantic_compat` and reorder factory**

Edit `src/raitap/transparency/factory.py`. The function currently starts at line 113:

```python
def check_explainer_visualiser_semantic_compat(
    explainer: object,
    explainer_target: str,
    visualisers: list[ConfiguredVisualiser],
) -> None:
    if not _requires_registry_semantics(explainer, explainer_target):
        return

    capability = explainer_capability(explainer)
    # ... rest of function
```

Replace the signature + the `capability` line:

```python
def check_explainer_visualiser_semantic_compat(
    explainer: object,
    explainer_target: str,
    visualisers: list[ConfiguredVisualiser],
    *,
    task_kind: TaskKind | None = None,
) -> None:
    if not _requires_registry_semantics(explainer, explainer_target):
        return

    capability = explainer_capability(explainer, task_kind=task_kind)
    # ... rest of function (unchanged)
```

(`TaskKind` import at the top of `factory.py`: `from raitap.types import TaskKind`.)

Next, reorder the callsite. Around line 178-183 the current code is:

```python
            check_explainer_visualiser_semantic_compat(
                explainer,
                explainer_target,
                visualisers,
            )
            # ... lines in between ...
            backend = _require_model_backend(model)
```

Reorder: resolve the backend first, then call pre-flight with `task_kind`:

```python
            backend = _require_model_backend(model)
            check_explainer_visualiser_semantic_compat(
                explainer,
                explainer_target,
                visualisers,
                task_kind=backend.task_kind,
            )
```

(Preserve any code that was between the two original calls — move it inside the new ordering as needed; if it referenced `backend` it now sees the same `backend` object.)

Also add the supported-output-space check inside `check_explainer_visualiser_semantic_compat` for cases where the visualiser declares `supported_output_spaces` non-empty. Search for the existing `supported_method_families` block (around line 130) and add an equivalent check right after it:

```python
        supported_output_spaces = _enum_frozenset(
            getattr(type(visualiser), "supported_output_spaces", frozenset()),
            ExplanationOutputSpace,
        )
        if supported_output_spaces and not capability.candidate_output_spaces.intersection(
            supported_output_spaces
        ):
            raise ValueError(
                f"Visualiser {type(visualiser).__name__!r} does not support explainer "
                f"output spaces {sorted(s.value for s in capability.candidate_output_spaces)}. "
                "Its supported output spaces are "
                f"{sorted(s.value for s in supported_output_spaces)}."
            )
```

(Reuse the existing `_enum_frozenset` helper. Import `ExplanationOutputSpace` at top of factory.py if not already.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/raitap/transparency/tests/test_factory.py -v`
Expected: all (existing + 2 new) pass.

- [ ] **Step 5: Transparency regression sweep**

Run: `uv run pytest src/raitap/transparency -x -q`
Expected: all pass.

- [ ] **Step 6: Pyright**

Run: `uv run pyright src/raitap/transparency/factory.py src/raitap/transparency/tests/test_factory.py`
Expected: 0 errors.

- [ ] **Step 7: Commit**

```bash
git add src/raitap/transparency/factory.py src/raitap/transparency/tests/test_factory.py
git commit -m "feat(transparency): factory backend-first + task-aware pre-flight

Resolve backend before check_explainer_visualiser_semantic_compat so
task_kind is available during semantic compatibility checks. Pre-flight
now reads candidate_output_spaces with task_kind threaded through. New
supported_output_spaces check rejects visualisers whose declared output
spaces don't intersect the candidate set. Issue #146 Phase 3 (A2 + D8)."
```

---

### Task 7: Typed `ForwardOutput` dataclass + callsite migrations (D23)

**Files:**
- Modify: `src/raitap/pipeline/outputs.py`
- Modify: `src/raitap/pipeline/phases/forward_pass.py`
- Modify: `src/raitap/pipeline/phases/prediction_summaries.py`
- Modify: `src/raitap/pipeline/phases/evaluate_metrics.py`
- Modify: `src/raitap/pipeline/phases/assess_robustness.py`
- Modify: `src/raitap/reporting/builder.py:1260`
- Create: `src/raitap/pipeline/tests/test_forward_output.py`
- Modify: `src/raitap/pipeline/tests/test_forward_pass.py` (if exists; else create test_forward_output.py covers it)

- [ ] **Step 1: Write the failing test for the dataclass**

Create `src/raitap/pipeline/tests/test_forward_output.py`:

```python
"""Tests for the typed ForwardOutput dataclass."""

from __future__ import annotations

import pytest
import torch

from raitap.pipeline.outputs import ForwardOutput
from raitap.types import TaskKind


def test_classification_forward_output_requires_predictions_tensor() -> None:
    with pytest.raises(ValueError, match="predictions_tensor"):
        ForwardOutput(task_kind=TaskKind.classification, batch_size=4)


def test_detection_forward_output_requires_detection_predictions() -> None:
    with pytest.raises(ValueError, match="detection_predictions"):
        ForwardOutput(task_kind=TaskKind.detection, batch_size=2)


def test_classification_forward_output_round_trip() -> None:
    predictions = torch.zeros(4, 10)
    out = ForwardOutput(
        task_kind=TaskKind.classification,
        batch_size=4,
        predictions_tensor=predictions,
    )
    assert out.task_kind is TaskKind.classification
    assert out.batch_size == 4
    assert torch.equal(out.predictions_tensor, predictions)
    assert out.detection_predictions is None


def test_detection_forward_output_round_trip() -> None:
    detection_predictions = [
        {
            "boxes": torch.zeros((1, 4)),
            "scores": torch.zeros(1),
            "labels": torch.zeros(1, dtype=torch.int64),
        },
        {
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros(0),
            "labels": torch.zeros(0, dtype=torch.int64),
        },
    ]
    out = ForwardOutput(
        task_kind=TaskKind.detection,
        batch_size=2,
        detection_predictions=detection_predictions,
    )
    assert out.task_kind is TaskKind.detection
    assert out.batch_size == 2
    assert out.detection_predictions == detection_predictions
    assert out.predictions_tensor is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/raitap/pipeline/tests/test_forward_output.py -v`
Expected: FAIL — `ForwardOutput` doesn't exist.

- [ ] **Step 3: Add the `ForwardOutput` dataclass + retype `RunOutputs.forward_output`**

Edit `src/raitap/pipeline/outputs.py`. Add the new dataclass just before `RunOutputs`:

```python
@dataclass(frozen=True)
class ForwardOutput:
    """Typed model forward output.

    Replaces the historical ``RunOutputs.forward_output: torch.Tensor`` so
    detection backends (whose forward produces ``list[dict[str, Tensor]]``)
    plug into the same downstream phases without overloading the tensor
    field. Classification path keeps the original tensor shape on
    :attr:`predictions_tensor`; detection path populates
    :attr:`detection_predictions`. :attr:`batch_size` is task-agnostic so
    reporting + UI callers don't need to branch.
    """

    task_kind: TaskKind
    batch_size: int
    predictions_tensor: torch.Tensor | None = None
    detection_predictions: list[dict[str, torch.Tensor]] | None = None

    def __post_init__(self) -> None:
        if self.task_kind is TaskKind.classification and self.predictions_tensor is None:
            raise ValueError(
                "ForwardOutput(task_kind=classification) requires predictions_tensor."
            )
        if self.task_kind is TaskKind.detection and self.detection_predictions is None:
            raise ValueError(
                "ForwardOutput(task_kind=detection) requires detection_predictions."
            )
```

Add the imports at the top of `outputs.py` if not present: `from raitap.types import TaskKind`, `import torch`.

Then change the existing `RunOutputs.forward_output: torch.Tensor` line to:

```python
    forward_output: ForwardOutput
```

- [ ] **Step 4: Update `forward_pass.py` to return `ForwardOutput`**

Edit `src/raitap/pipeline/phases/forward_pass.py`. Find the existing `forward_pass` function (its body chunks `data` through `backend(...)` and returns a tensor). Wrap the return:

```python
def forward_pass(
    config: AppConfig,
    data: Data,
    backend: ModelBackend,
) -> ForwardOutput:
    chunked_outputs = ...   # existing chunked-loop logic stays
    if backend.task_kind is TaskKind.detection:
        detection_predictions = _flatten_detection_chunks(chunked_outputs)
        return ForwardOutput(
            task_kind=TaskKind.detection,
            batch_size=len(detection_predictions),
            detection_predictions=detection_predictions,
        )

    predictions_tensor = extract_primary_tensor(chunked_outputs, ...)
    return ForwardOutput(
        task_kind=TaskKind.classification,
        batch_size=int(predictions_tensor.shape[0]),
        predictions_tensor=predictions_tensor,
    )
```

Add helper:

```python
def _flatten_detection_chunks(
    chunks: list[list[dict[str, torch.Tensor]]],
) -> list[dict[str, torch.Tensor]]:
    """Concatenate per-chunk detection lists into a single length-N list."""
    flat: list[dict[str, torch.Tensor]] = []
    for chunk in chunks:
        flat.extend(chunk)
    return flat
```

(The exact body of `forward_pass` depends on the current implementation. Read `pipeline/phases/forward_pass.py` once before editing; preserve the chunked-iteration logic verbatim, only wrap the return value.)

Import `TaskKind` + `ForwardOutput` at top of file:

```python
from raitap.types import TaskKind
from raitap.pipeline.outputs import ForwardOutput
```

- [ ] **Step 5: Migrate the 5 consumer callsites**

5a. `prediction_summaries.py:30` — replace the `forward_output.ndim != 2 or forward_output.shape[1] < 2` guard with a `task_kind` check:

```python
def prediction_summaries(
    forward_output: ForwardOutput,
    sample_ids: Sequence[str] | None,
    targets: torch.Tensor | None,
) -> tuple[PredictionSummary, ...]:
    if forward_output.task_kind is not TaskKind.classification:
        # Detection / regression / etc. don't have a "predicted class +
        # confidence" concept per sample.
        return ()
    predictions_tensor = forward_output.predictions_tensor
    assert predictions_tensor is not None
    if predictions_tensor.ndim != 2 or predictions_tensor.shape[1] < 2:
        return ()
    # ... existing softmax + argmax body unchanged, just read from
    # predictions_tensor instead of the raw arg.
```

5b. `evaluate_metrics.py:41-44` — accept `ForwardOutput`, dispatch on task_kind:

```python
def evaluate_metrics(
    config: AppConfig,
    forward_output: ForwardOutput,
    labels: torch.Tensor | list[dict[str, torch.Tensor]] | None,
) -> MetricsEvaluation | None:
    if not metrics_run_enabled(config):
        return None
    assert config.metrics is not None

    raitap_log.info("Computing metrics...")

    if forward_output.task_kind is TaskKind.detection:
        # Detection: predictions are list[dict]; targets are list[dict] from D22 loader.
        return Metrics(
            config,
            forward_output.detection_predictions,
            labels,
        )

    # Classification — existing body unchanged except for reading
    # predictions_tensor instead of the raw arg.
    predictions_tensor = forward_output.predictions_tensor
    assert predictions_tensor is not None
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

5c. `assess_robustness.py:38` — replace `ndim != 2 or shape[1] < 2` with task guard:

```python
def assess_robustness(...):
    ...
    if forward_output.task_kind is not TaskKind.classification:
        # Robustness against detection models is handled by Phase 4
        # (DetectionAdversarialLoss). Empirical / formal robustness in this
        # phase only supports classification today.
        return None
    predictions_tensor = forward_output.predictions_tensor
    assert predictions_tensor is not None
    if predictions_tensor.ndim != 2 or predictions_tensor.shape[1] < 2:
        return None
    # ... rest unchanged, read from predictions_tensor.
```

5d. `reporting/builder.py:1260` — replace tensor introspection with `.batch_size`:

```python
def _batch_size_from_run_outputs(outputs: RunOutputs) -> int:
    return outputs.forward_output.batch_size
```

(or inline the equivalent — preserve the surrounding function name; just change the two lines.)

- [ ] **Step 6: Run the dataclass tests + regression sweep**

Run: `uv run pytest src/raitap/pipeline/tests/test_forward_output.py -v`
Expected: 4 pass.

Run: `uv run pytest src/raitap -x -q`
Expected: all existing tests pass — classification code path migrated cleanly.

If any classification test fails, fix the migrated callsite to preserve original behaviour exactly (no new branches; only the indirection through `ForwardOutput.predictions_tensor`).

- [ ] **Step 7: Pyright on the modified files**

Run: `uv run pyright src/raitap/pipeline/outputs.py src/raitap/pipeline/phases/forward_pass.py src/raitap/pipeline/phases/prediction_summaries.py src/raitap/pipeline/phases/evaluate_metrics.py src/raitap/pipeline/phases/assess_robustness.py src/raitap/reporting/builder.py src/raitap/pipeline/tests/test_forward_output.py`
Expected: 0 errors.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor(pipeline): typed ForwardOutput replaces tensor-only field

ForwardOutput dataclass carries task_kind + batch_size plus either
predictions_tensor (classification) or detection_predictions (detection).
Six consumer callsites migrate: forward_pass (returns ForwardOutput),
prediction_summaries / evaluate_metrics / assess_robustness (guard on
task_kind, read predictions_tensor), builder._batch_size_from_run_outputs
(reads .batch_size). Classification path bit-for-bit identical. Issue
#146 Phase 3 (D23)."
```

---

### Task 8: `Data._load_detection_labels()` codepath (D22)

**Files:**
- Modify: `src/raitap/data/data.py`
- Create: `src/raitap/data/tests/test_detection_labels.py`

- [ ] **Step 1: Write the failing tests**

Create `src/raitap/data/tests/test_detection_labels.py`:

```python
"""Tests for Data._load_detection_labels — list[dict] per-sample boxes + labels."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from raitap.data.data import Data
from raitap.configs.schema import AppConfig


def _write_detection_labels_json(path: Path) -> None:
    payload = [
        {
            "sample_id": "img_0",
            "boxes": [[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 12.0, 12.0]],
            "labels": [1, 2],
        },
        {
            "sample_id": "img_1",
            "boxes": [],
            "labels": [],
        },
        {
            "sample_id": "img_2",
            "boxes": [[3.0, 3.0, 6.0, 6.0]],
            "labels": [1],
        },
    ]
    path.write_text(json.dumps(payload))


def test_load_detection_labels_returns_list_of_dicts(tmp_path) -> None:
    labels_path = tmp_path / "boxes.json"
    _write_detection_labels_json(labels_path)
    cfg = AppConfig(...)  # construct minimal AppConfig with data.labels.source = labels_path,
                          # data.labels.kind = "detection"
    cfg.data.labels.source = str(labels_path)
    cfg.data.labels.kind = "detection"

    data = Data.__new__(Data)
    out = data._load_detection_labels(cfg)
    assert isinstance(out, list)
    assert len(out) == 3
    assert torch.equal(out[0]["boxes"], torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 12.0, 12.0]]))
    assert torch.equal(out[0]["labels"], torch.tensor([1, 2], dtype=torch.int64))
    assert out[1]["boxes"].shape == (0, 4)
    assert out[1]["labels"].shape == (0,)


def test_load_detection_labels_rejects_mismatched_box_label_counts(tmp_path) -> None:
    labels_path = tmp_path / "bad.json"
    labels_path.write_text(json.dumps([
        {"sample_id": "x", "boxes": [[0.0, 0.0, 1.0, 1.0]], "labels": [1, 2]},
    ]))
    cfg = AppConfig(...)
    cfg.data.labels.source = str(labels_path)
    cfg.data.labels.kind = "detection"

    data = Data.__new__(Data)
    with pytest.raises(ValueError, match="boxes and labels"):
        data._load_detection_labels(cfg)


def test_load_labels_returns_none_when_no_labels_configured(tmp_path) -> None:
    cfg = AppConfig(...)
    cfg.data.labels = None
    data = Data.__new__(Data)
    assert data._load_detection_labels(cfg) is None
```

(Note: `AppConfig(...)` placeholder above must be replaced with whatever the minimum-viable `AppConfig` construction looks like in the repo's existing tests — check `src/raitap/data/tests/test_data.py` for the existing pattern and copy it. Don't invent fields.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/raitap/data/tests/test_detection_labels.py -v`
Expected: FAIL — `_load_detection_labels` doesn't exist.

- [ ] **Step 3: Add `_load_detection_labels` + discriminator on `Data`**

Edit `src/raitap/data/data.py`. Add the new method right after `_load_labels`:

```python
def _load_detection_labels(
    self, cfg: AppConfig
) -> list[dict[str, torch.Tensor]] | None:
    """Load per-sample detection targets (boxes + labels).

    Expected on-disk shape: JSON file (list of records) with each record
    carrying ``sample_id`` (str), ``boxes`` (list of [x1, y1, x2, y2]
    floats), and ``labels`` (list of ints). Returns a list of length N
    where N matches the dataset's sample count; each entry is a dict with
    ``boxes: (M_i, 4) float32`` and ``labels: (M_i,) int64`` tensors.
    Samples with no boxes get shape-(0, 4) / shape-(0,) tensors.

    Returns ``None`` when ``data.labels`` is not configured. Discriminated
    by ``data.labels.kind: detection``; ``_load_labels`` continues to
    handle classification (the default).
    """
    labels_cfg = _get_optional_config_value(cfg.data, "labels")
    labels_source = _get_optional_config_value(labels_cfg, "source")
    if not labels_source:
        return None

    labels_path = get_source_path(labels_source, kind=SourceKind.LABELS)
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Detection labels file not found at {labels_path}."
        )

    with labels_path.open() as fh:
        records = json.load(fh)
    if not isinstance(records, list):
        raise ValueError(
            f"Detection labels file {labels_path} must be a JSON array."
        )

    out: list[dict[str, torch.Tensor]] = []
    for index, record in enumerate(records):
        boxes_raw = record.get("boxes", [])
        labels_raw = record.get("labels", [])
        if len(boxes_raw) != len(labels_raw):
            raise ValueError(
                f"Sample index {index}: boxes and labels must have matching "
                f"length (got {len(boxes_raw)} boxes vs {len(labels_raw)} labels)."
            )
        boxes_tensor = (
            torch.tensor(boxes_raw, dtype=torch.float32)
            if boxes_raw
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        labels_tensor = (
            torch.tensor(labels_raw, dtype=torch.int64)
            if labels_raw
            else torch.zeros((0,), dtype=torch.int64)
        )
        if boxes_tensor.ndim != 2 or boxes_tensor.shape[1] != 4:
            raise ValueError(
                f"Sample index {index}: boxes must be shape (M_i, 4); got "
                f"{tuple(boxes_tensor.shape)}."
            )
        out.append({"boxes": boxes_tensor, "labels": labels_tensor})

    return out
```

Add the `import json` at the top of `data.py` if not already present.

Then add a discriminator at the existing `_load_labels` callsite (where `Data.__init__` decides what to load). Find that callsite and update it:

```python
labels_kind = _get_optional_config_value(
    _get_optional_config_value(cfg.data, "labels"), "kind"
)
if labels_kind == "detection":
    self.labels = self._load_detection_labels(cfg)
else:
    self.labels = self._load_labels(cfg)
```

`Data.labels` type annotation must be loosened: `torch.Tensor | list[dict[str, torch.Tensor]] | None`. Update the class-level annotation.

If `LabelsConfig` (or whatever the existing schema piece is) doesn't have a `kind: str | None = None` field, add it. Look at `src/raitap/configs/schema.py` and find the labels config dataclass; add:

```python
@dataclass
class LabelsConfig:
    source: str | None = MISSING
    id_column: str | None = None
    column: str | None = None
    encoding: str | None = None
    id_strategy: str | None = None
    kind: str | None = None   # NEW — "detection" opts into _load_detection_labels
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/raitap/data/tests/test_detection_labels.py -v`
Expected: 3 pass.

- [ ] **Step 5: Data regression sweep**

Run: `uv run pytest src/raitap/data -x -q`
Expected: all existing tests pass (classification path untouched).

- [ ] **Step 6: Pyright**

Run: `uv run pyright src/raitap/data/data.py src/raitap/data/tests/test_detection_labels.py src/raitap/configs/schema.py`
Expected: 0 errors.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(data): detection labels loader

New Data._load_detection_labels() returns list[dict] per-sample boxes +
labels (float32 xyxy + int64 labels, shape-(0,4) for empty samples). Opt
in via data.labels.kind=detection in YAML. _load_labels stays
classification-only. LabelsConfig schema gains optional kind field.
Issue #146 Phase 3 (D22)."
```

---

### Task 9: `evaluate_metrics` detection dispatch verified end-to-end (D18′)

**Note:** Task 7 already migrated `evaluate_metrics` to accept `ForwardOutput`. Task 9 verifies the detection branch works with real `DetectionMetrics`.

**Files:**
- Modify: `src/raitap/metrics/tests/test_factory_evaluate.py` — append (or create `test_detection_evaluate.py` if cleaner)

- [ ] **Step 1: Write the failing test**

Append (or create):

```python
def test_evaluate_metrics_runs_detection_metrics_on_list_of_dicts() -> None:
    import torch
    from raitap.configs.schema import AppConfig
    from raitap.metrics import Metrics, MetricsEvaluation
    from raitap.pipeline.outputs import ForwardOutput
    from raitap.pipeline.phases.evaluate_metrics import evaluate_metrics
    from raitap.types import TaskKind

    predictions = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1], dtype=torch.int64),
        },
    ]
    labels = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([1], dtype=torch.int64),
        },
    ]
    cfg = AppConfig(...)  # configure cfg.metrics._target_ = "DetectionMetrics" + minimal scaffold
    forward_output = ForwardOutput(
        task_kind=TaskKind.detection,
        batch_size=1,
        detection_predictions=predictions,
    )

    result = evaluate_metrics(cfg, forward_output, labels)
    assert isinstance(result, MetricsEvaluation)
    # mAP scalars exposed under the standard torchmetrics keys.
    metrics_dict = result.metrics  # MetricsEvaluation.metrics is dict[str, float]
    assert any(key.lower().startswith("map") for key in metrics_dict)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest src/raitap/metrics/tests/test_factory_evaluate.py -v -k detection_metrics_on_list_of_dicts`
Expected: FAIL — `Metrics()` constructed with classification kwargs may still call shape-checking helpers that reject list[dict].

- [ ] **Step 3: Verify `Metrics(...)` accepts list[dict]**

Open `src/raitap/metrics/__init__.py` and the `Metrics` class. If `Metrics.__init__` enforces tensor-shape on predictions OR `resolve_metric_targets` rejects list[dict] targets, branch on the predictions type:

```python
class Metrics:
    def __init__(self, cfg, predictions, targets):
        if isinstance(predictions, list):
            # Detection — delegate directly to the configured metric adapter
            # without classification's argmax/probs handling.
            adapter = _instantiate_metric_adapter(cfg)
            adapter.update(predictions, targets)
            self.metrics, self.artifacts = adapter.compute().to_pair()
            return
        # Existing classification path unchanged.
        ...
```

(The exact placement depends on the current `Metrics.__init__` body. Read it first, preserve everything except inserting the early-return list branch.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/raitap/metrics -x -q`
Expected: all (existing + new) pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(metrics): evaluate_metrics dispatches DetectionMetrics for list[dict]

Metrics class branches early when predictions are a list[dict]
(detection torchvision format) — delegates straight to the configured
metric adapter without classification's argmax/probs handling. With
ForwardOutput.detection_predictions + D22 detection label loader,
DetectionMetrics now runs end-to-end and exposes mAP scalars. Issue #146
Phase 3 (D18′)."
```

---

### Task 10: `pipeline/phases/explain_detection.py` — the K-loop (D24 + A1 + D21 + D10 + D13)

**Files:**
- Create: `src/raitap/pipeline/phases/explain_detection.py`
- Create: `src/raitap/pipeline/phases/tests/test_explain_detection.py`
- Modify: `src/raitap/pipeline/phases/assess_transparency.py` — delegate when detection

- [ ] **Step 1: Write the failing test**

Create `src/raitap/pipeline/phases/tests/test_explain_detection.py`:

```python
"""Tests for the detection explain phase."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from raitap.models.backend import TorchBackend
from raitap.models.task_wrappers import DetectionTarget, ScalarDetectionWrapper
from raitap.pipeline.outputs import ForwardOutput
from raitap.pipeline.phases.explain_detection import explain_detection
from raitap.transparency.contracts import DetectionBox
from raitap.types import TaskKind


class _FakeDetector(nn.Module):
    """Detector that always returns the same per-sample list[dict].

    Forward output (per batch sample):
      boxes  = [[0, 0, 10, 10], [50, 50, 60, 60], [5, 5, 15, 15]]
      scores = [0.9, 0.4, 0.7]
      labels = [1, 2, 1]
    """

    def forward(self, images):
        bs = images.shape[0]
        return [
            {
                "boxes": torch.tensor(
                    [[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0], [5.0, 5.0, 15.0, 15.0]]
                ),
                "scores": torch.tensor([0.9, 0.4, 0.7]),
                "labels": torch.tensor([1, 2, 1]),
            }
            for _ in range(bs)
        ]


class _RecordingExplainer:
    """Test double — captures every explain() invocation."""

    algorithm = "IntegratedGradients"
    algorithm_registry = {"IntegratedGradients": frozenset()}
    output_payload_kind = None
    output_scope = None

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def explain(
        self,
        model,
        inputs,
        *,
        backend=None,
        run_dir=None,
        explainer_target=None,
        explainer_name=None,
        visualisers=None,
        raitap_kwargs=None,
        **kwargs,
    ):
        from raitap.transparency.contracts import (
            ExplanationOutputSpace,
            ExplanationPayloadKind,
            ExplanationScope,
            InputKind,
            InputSpec,
            OutputSpaceSpec,
            ScopeDefinitionStep,
            TensorLayout,
        )
        from raitap.transparency.contracts import ExplanationSemantics
        from raitap.transparency.results import ExplanationResult

        self.calls.append(
            {
                "model": model,
                "inputs_shape": tuple(inputs.shape),
                "run_dir": run_dir,
                "call_target": kwargs.get("target"),
            }
        )
        semantics = ExplanationSemantics(
            scope=ExplanationScope.LOCAL,
            scope_definition_step=ScopeDefinitionStep.EXPLAINER_OUTPUT,
            payload_kind=ExplanationPayloadKind.ATTRIBUTIONS,
            method_families=frozenset(),
            target=None,
            sample_selection=None,
            input_spec=InputSpec(
                kind=InputKind.IMAGE,
                shape=tuple(inputs.shape),
                layout=TensorLayout.BATCH_CHANNEL_HEIGHT_WIDTH,
            ),
            output_space=OutputSpaceSpec(
                space=ExplanationOutputSpace.DETECTION_BOXES,
                shape=tuple(inputs.shape),
                layout=TensorLayout.BATCH_CHANNEL_HEIGHT_WIDTH,
            ),
        )
        return ExplanationResult(
            attributions=torch.zeros_like(inputs),
            inputs=inputs,
            run_dir=run_dir,
            experiment_name=None,
            explainer_target=explainer_target,
            algorithm=self.algorithm,
            explainer_name=explainer_name,
            visualisers=[],
            semantics=semantics,
        )


@pytest.fixture
def detection_forward_output() -> ForwardOutput:
    return ForwardOutput(
        task_kind=TaskKind.detection,
        batch_size=2,
        detection_predictions=[
            {
                "boxes": torch.tensor(
                    [[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0], [5.0, 5.0, 15.0, 15.0]]
                ),
                "scores": torch.tensor([0.9, 0.4, 0.7]),
                "labels": torch.tensor([1, 2, 1]),
            },
            {
                "boxes": torch.zeros((0, 4)),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.int64),
            },
        ],
    )


def test_explain_detection_filters_below_threshold_and_caps_at_max_boxes(
    detection_forward_output, tmp_path
) -> None:
    backend = TorchBackend(_FakeDetector(), task_kind=TaskKind.detection)
    inputs = torch.zeros(2, 3, 8, 8)
    explainer = _RecordingExplainer()
    raitap_cfg = {"detection": {"score_threshold": 0.5, "max_boxes": 5}}

    results = list(
        explain_detection(
            inputs=inputs,
            forward_output=detection_forward_output,
            backend=backend,
            explainer=explainer,
            explainer_target="t",
            explainer_name="x",
            visualisers=[],
            base_run_dir=tmp_path,
            raitap_kwargs=raitap_cfg,
            call_kwargs={},
        )
    )

    # Sample 0: 3 detections (0.9, 0.4, 0.7) → filter ≥ 0.5 → keep (0.9, 0.7) → 2 boxes
    # Sample 1: 0 detections → 0 boxes
    assert len(results) == 2
    assert len(explainer.calls) == 2

    # Display ranking is by score descending. Raw indices preserved: 0.9 is raw_index=0,
    # 0.7 is raw_index=2.
    boxes = [r.detection_box for r in results]
    assert boxes[0].display_index == 0
    assert boxes[0].raw_index == 0
    assert boxes[0].score == pytest.approx(0.9)
    assert boxes[1].display_index == 1
    assert boxes[1].raw_index == 2
    assert boxes[1].score == pytest.approx(0.7)


def test_explain_detection_passes_target_zero_overriding_call_kwargs(
    detection_forward_output, tmp_path
) -> None:
    backend = TorchBackend(_FakeDetector(), task_kind=TaskKind.detection)
    explainer = _RecordingExplainer()

    list(
        explain_detection(
            inputs=torch.zeros(2, 3, 8, 8),
            forward_output=detection_forward_output,
            backend=backend,
            explainer=explainer,
            explainer_target="t",
            explainer_name="x",
            visualisers=[],
            base_run_dir=tmp_path,
            raitap_kwargs={"detection": {"score_threshold": 0.5, "max_boxes": 5}},
            call_kwargs={"target": 7},     # user wrote target=7; phase must override to 0
        )
    )

    for call in explainer.calls:
        assert call["call_target"] == 0


def test_explain_detection_rejects_auto_pred(detection_forward_output, tmp_path) -> None:
    from raitap.utils.errors import RaitapError

    backend = TorchBackend(_FakeDetector(), task_kind=TaskKind.detection)
    explainer = _RecordingExplainer()

    with pytest.raises(RaitapError, match="auto_pred"):
        list(
            explain_detection(
                inputs=torch.zeros(2, 3, 8, 8),
                forward_output=detection_forward_output,
                backend=backend,
                explainer=explainer,
                explainer_target="t",
                explainer_name="x",
                visualisers=[],
                base_run_dir=tmp_path,
                raitap_kwargs={"detection": {"score_threshold": 0.5, "max_boxes": 5}},
                call_kwargs={"target": "auto_pred"},
            )
        )


def test_explain_detection_skips_samples_with_no_passing_boxes(
    detection_forward_output, tmp_path, caplog
) -> None:
    backend = TorchBackend(_FakeDetector(), task_kind=TaskKind.detection)
    explainer = _RecordingExplainer()

    # score_threshold above max score for sample 0 (0.9) → both samples produce 0 boxes
    results = list(
        explain_detection(
            inputs=torch.zeros(2, 3, 8, 8),
            forward_output=detection_forward_output,
            backend=backend,
            explainer=explainer,
            explainer_target="t",
            explainer_name="x",
            visualisers=[],
            base_run_dir=tmp_path,
            raitap_kwargs={"detection": {"score_threshold": 0.99, "max_boxes": 5}},
            call_kwargs={},
        )
    )
    assert results == []


def test_explain_detection_per_box_run_dir_layout(
    detection_forward_output, tmp_path
) -> None:
    backend = TorchBackend(_FakeDetector(), task_kind=TaskKind.detection)
    explainer = _RecordingExplainer()

    list(
        explain_detection(
            inputs=torch.zeros(2, 3, 8, 8),
            forward_output=detection_forward_output,
            backend=backend,
            explainer=explainer,
            explainer_target="t",
            explainer_name="my_explainer",
            visualisers=[],
            base_run_dir=tmp_path,
            raitap_kwargs={"detection": {"score_threshold": 0.5, "max_boxes": 5}},
            call_kwargs={},
        )
    )

    # Expect sample_0/box_0 and sample_0/box_2 (raw indices of the kept boxes).
    expected_dirs = {
        tmp_path / "sample_0" / "box_0",
        tmp_path / "sample_0" / "box_2",
    }
    actual_dirs = {call["run_dir"] for call in explainer.calls}
    assert actual_dirs == expected_dirs


def test_explain_detection_attaches_detection_box_and_original_sample_index(
    detection_forward_output, tmp_path
) -> None:
    backend = TorchBackend(_FakeDetector(), task_kind=TaskKind.detection)
    explainer = _RecordingExplainer()

    results = list(
        explain_detection(
            inputs=torch.zeros(2, 3, 8, 8),
            forward_output=detection_forward_output,
            backend=backend,
            explainer=explainer,
            explainer_target="t",
            explainer_name="x",
            visualisers=[],
            base_run_dir=tmp_path,
            raitap_kwargs={"detection": {"score_threshold": 0.5, "max_boxes": 5}},
            call_kwargs={},
        )
    )

    assert all(isinstance(r.detection_box, DetectionBox) for r in results)
    assert all(r.original_sample_index == 0 for r in results)
    assert results[0].detection_box.xyxy == (0.0, 0.0, 10.0, 10.0)
    assert results[0].detection_box.label_index == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/raitap/pipeline/phases/tests/test_explain_detection.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Create the detection phase**

Create `src/raitap/pipeline/phases/explain_detection.py`:

```python
"""Detection-task transparency phase — one ExplanationResult per detected box.

Issue #146 Phase 3. Reads pre-computed detection predictions from
:class:`ForwardOutput` (D24, no second forward pass), filters per sample by
``score_threshold`` then top ``max_boxes``, and calls the explainer K times
per sample with a faithful ``DetectionTarget(mode="reference_match", ...)``
anchored to that box's xyxy + label.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import torch

from raitap import raitap_log
from raitap.models.task_wrappers import DetectionTarget, ScalarDetectionWrapper
from raitap.transparency.contracts import DetectionBox
from raitap.types import TaskKind
from raitap.utils.errors import RaitapError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from raitap.models.backend import ModelBackend
    from raitap.pipeline.outputs import ForwardOutput
    from raitap.transparency.results import ConfiguredVisualiser, ExplanationResult


_DEFAULT_SCORE_THRESHOLD = 0.5
_DEFAULT_MAX_BOXES = 5
_DEFAULT_IOU_THRESHOLD = 0.5


def explain_detection(
    *,
    inputs: torch.Tensor,
    forward_output: ForwardOutput,
    backend: ModelBackend,
    explainer: Any,
    explainer_target: str,
    explainer_name: str,
    visualisers: Sequence[ConfiguredVisualiser],
    base_run_dir: Path,
    raitap_kwargs: dict[str, Any] | None,
    call_kwargs: dict[str, Any],
) -> Iterator[ExplanationResult]:
    """Yield one ExplanationResult per detected box.

    Reads pre-computed detection predictions from
    ``forward_output.detection_predictions`` (no second backend call).
    Filters each sample by ``score_threshold`` then keeps top
    ``max_boxes`` by score descending; loops the explainer K times per
    sample with reference-match targets.
    """
    if forward_output.task_kind is not TaskKind.detection:
        raise RaitapError(
            "explain_detection invoked on a non-detection ForwardOutput "
            f"(task_kind={forward_output.task_kind!r})."
        )

    detection_predictions = forward_output.detection_predictions
    assert detection_predictions is not None  # invariant from ForwardOutput.__post_init__

    detection_cfg = (raitap_kwargs or {}).get("detection", {})
    score_threshold = float(detection_cfg.get("score_threshold", _DEFAULT_SCORE_THRESHOLD))
    max_boxes = int(detection_cfg.get("max_boxes", _DEFAULT_MAX_BOXES))
    iou_threshold = float(detection_cfg.get("iou_threshold", _DEFAULT_IOU_THRESHOLD))

    if max_boxes < 1:
        raise RaitapError(f"raitap.detection.max_boxes must be >= 1; got {max_boxes!r}.")
    if not 0.0 <= iou_threshold <= 1.0:
        raise RaitapError(
            f"raitap.detection.iou_threshold must lie in [0, 1]; got {iou_threshold!r}."
        )

    # D21 — target normalisation: wrapper exposes one scalar channel.
    requested_target = call_kwargs.get("target")
    if requested_target == "auto_pred":
        raise RaitapError(
            "config.transparency.<explainer>.call.target=auto_pred is not supported "
            "for detection tasks: the ScalarDetectionWrapper exposes a single scalar "
            "channel, so argmax over it always returns 0. Set call.target=0 explicitly."
        )
    if requested_target is not None and requested_target != 0:
        raitap_log.warn(
            f"Overriding call.target={requested_target!r} to 0 for detection task "
            "(wrapper exposes a single scalar channel)."
        )
    normalised_call_kwargs = dict(call_kwargs)
    normalised_call_kwargs["target"] = 0

    base_model = backend.as_model_for_explanation()

    for sample_index, predictions_i in enumerate(detection_predictions):
        # A1 — raw-index correct filter.
        scores = predictions_i.get("scores", torch.zeros(0))
        boxes = predictions_i.get("boxes", torch.zeros((0, 4)))
        labels = predictions_i.get("labels", torch.zeros(0, dtype=torch.int64))

        if scores.numel() == 0:
            raitap_log.warn(
                f"sample_index={sample_index}: 0 detections from forward pass; "
                "emitting no detection explanations."
            )
            continue

        mask = scores >= score_threshold
        raw_candidates = torch.nonzero(mask, as_tuple=False).flatten()
        if raw_candidates.numel() == 0:
            raitap_log.warn(
                f"sample_index={sample_index}: 0 boxes passed "
                f"score_threshold={score_threshold!r}; "
                f"max_score={float(scores.max())!r}; "
                "emitting no detection explanations."
            )
            continue

        order = scores[raw_candidates].argsort(descending=True)
        top_k_raw_indices = raw_candidates[order[:max_boxes]]

        sample_inputs = inputs[sample_index : sample_index + 1]

        for display_index, raw_index_tensor in enumerate(top_k_raw_indices.tolist()):
            raw_index = int(raw_index_tensor)
            reference_xyxy_tensor = boxes[raw_index]
            reference_xyxy = tuple(float(v) for v in reference_xyxy_tensor.tolist())
            reference_label = int(labels[raw_index].item())
            score = float(scores[raw_index].item())

            target = DetectionTarget(
                mode="reference_match",
                reference_xyxy=reference_xyxy,
                reference_label=reference_label,
                iou_threshold=iou_threshold,
            )
            wrapped = ScalarDetectionWrapper(base_model, target=target)

            per_box_run_dir = base_run_dir / f"sample_{sample_index}" / f"box_{raw_index}"
            per_box_run_dir.mkdir(parents=True, exist_ok=True)

            result = explainer.explain(
                wrapped,
                sample_inputs,
                backend=backend,
                run_dir=per_box_run_dir,
                explainer_target=explainer_target,
                explainer_name=explainer_name,
                visualisers=list(visualisers),
                raitap_kwargs=raitap_kwargs,
                **normalised_call_kwargs,
            )
            result.detection_box = DetectionBox(
                display_index=display_index,
                raw_index=raw_index,
                xyxy=reference_xyxy,
                score=score,
                label_index=reference_label,
                label_name=None,
            )
            result.original_sample_index = sample_index

            yield result
```

- [ ] **Step 4: Wire `explain_detection` into `assess_transparency`**

Edit `src/raitap/pipeline/phases/assess_transparency.py`. After the existing per-explainer iteration, add a branch on `backend.task_kind`. Find the existing body that calls `explainer.explain(...)` once per explainer; before that call, check the task:

```python
if backend.task_kind is TaskKind.detection:
    from raitap.pipeline.phases.explain_detection import explain_detection
    results = list(explain_detection(
        inputs=inputs,
        forward_output=forward_output,
        backend=backend,
        explainer=explainer,
        explainer_target=explainer_target,
        explainer_name=explainer_name,
        visualisers=visualisers,
        base_run_dir=resolve_run_dir(config, subdir=f"transparency/{explainer_name}"),
        raitap_kwargs=raitap_cfg,
        call_kwargs=merged_call_kwargs,
    ))
    for r in results:
        explanations.append(r)
    continue       # skip the classification single-call path below

# Existing classification path — explainer.explain(...) once, append result.
result = explainer.explain(...)
explanations.append(result)
```

(Read `assess_transparency.py` before editing — the exact local variable names may differ. Preserve everything else. `forward_output` must already be passed into the phase; if it isn't yet, thread it through the orchestrator.)

Update `assess_transparency` signature to accept `forward_output: ForwardOutput` if not already, and have the orchestrator pass it in.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest src/raitap/pipeline/phases/tests/test_explain_detection.py -v`
Expected: 6 pass.

Run: `uv run pytest src/raitap/pipeline -x -q`
Expected: all (existing + new) pass.

- [ ] **Step 6: Pyright**

Run: `uv run pyright src/raitap/pipeline/phases/explain_detection.py src/raitap/pipeline/phases/assess_transparency.py src/raitap/pipeline/phases/tests/test_explain_detection.py`
Expected: 0 errors.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(pipeline): detection explain phase — one result per box

Reads pre-computed detection predictions from
ForwardOutput.detection_predictions (no second forward pass). Per
sample, filters by score_threshold (default 0.5), keeps top max_boxes
(default 5) by score with correct raw-index masking. Per kept box:
constructs ScalarDetectionWrapper with reference_match target
(IoU+label anchored), per-box run_dir
sample_{i}/box_{raw_index}/, target=0 normalisation,
auto_pred rejection. Attaches DetectionBox + original_sample_index to
the result. Empty detections skip + warn. Issue #146 Phase 3
(D10/D13/D21/D24, A1 raw-index fix)."
```

---

### Task 11: `DetectionImageVisualiser`

**Files:**
- Create: `src/raitap/transparency/visualisers/detection_image_visualiser.py`
- Create: `src/raitap/transparency/tests/test_detection_image_visualiser.py`
- Modify: `src/raitap/transparency/visualisers/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `src/raitap/transparency/tests/test_detection_image_visualiser.py`:

```python
"""Tests for DetectionImageVisualiser."""

from __future__ import annotations

import pytest
import torch

from raitap.transparency.contracts import (
    DetectionBox,
    ExplanationOutputSpace,
    ExplanationPayloadKind,
    ExplanationScope,
    VisualisationContext,
)
from raitap.transparency.visualisers.detection_image_visualiser import (
    DetectionImageVisualiser,
)
from raitap.types import TaskKind


def _box(display_index=0, raw_index=0, label_name=None) -> DetectionBox:
    return DetectionBox(
        display_index=display_index,
        raw_index=raw_index,
        xyxy=(2.0, 3.0, 22.0, 23.0),
        score=0.87,
        label_index=4,
        label_name=label_name,
    )


def test_visualiser_supports_detection_output_space_only() -> None:
    assert DetectionImageVisualiser.supported_output_spaces == frozenset(
        {ExplanationOutputSpace.DETECTION_BOXES}
    )
    assert DetectionImageVisualiser.supported_payload_kinds == frozenset(
        {ExplanationPayloadKind.ATTRIBUTIONS}
    )
    assert DetectionImageVisualiser.supported_scopes == frozenset(
        {ExplanationScope.LOCAL}
    )
    assert DetectionImageVisualiser.supported_tasks == frozenset(
        {TaskKind.detection}
    )
    assert DetectionImageVisualiser.embeds_original_input is True


def test_visualiser_returns_figure_with_one_panel_and_axis_limits_match_image() -> None:
    vis = DetectionImageVisualiser()
    attributions = torch.zeros(1, 3, 64, 64)
    inputs = torch.rand(1, 3, 64, 64)
    ctx = VisualisationContext(
        algorithm="IntegratedGradients",
        sample_names=["img_0"],
        show_sample_names=False,
        detection_box=_box(label_name="car"),
    )

    fig = vis.visualise(attributions, inputs, context=ctx)
    axes = fig.get_axes()
    # one image panel; we don't mandate colorbar count
    assert len(axes) >= 1
    main_ax = axes[0]
    xlim = main_ax.get_xlim()
    ylim = main_ax.get_ylim()
    assert xlim[1] - xlim[0] == pytest.approx(64.0, abs=2.0)
    assert abs(ylim[1] - ylim[0]) == pytest.approx(64.0, abs=2.0)


def test_visualiser_title_carries_label_name_and_score() -> None:
    vis = DetectionImageVisualiser()
    inputs = torch.rand(1, 3, 32, 32)
    attributions = torch.zeros_like(inputs)
    ctx = VisualisationContext(
        algorithm="x", sample_names=None, show_sample_names=False,
        detection_box=_box(label_name="car"),
    )
    fig = vis.visualise(attributions, inputs, context=ctx)
    main_ax = fig.get_axes()[0]
    title = main_ax.get_title()
    assert "car" in title
    assert "0.87" in title or "0.9" in title  # allow rounding


def test_visualiser_falls_back_to_class_id_when_label_name_missing() -> None:
    vis = DetectionImageVisualiser()
    inputs = torch.rand(1, 3, 32, 32)
    attributions = torch.zeros_like(inputs)
    ctx = VisualisationContext(
        algorithm="x", sample_names=None, show_sample_names=False,
        detection_box=_box(label_name=None),
    )
    fig = vis.visualise(attributions, inputs, context=ctx)
    title = fig.get_axes()[0].get_title()
    assert "class 4" in title


def test_visualiser_raises_when_detection_box_missing() -> None:
    vis = DetectionImageVisualiser()
    inputs = torch.rand(1, 3, 32, 32)
    attributions = torch.zeros_like(inputs)
    ctx = VisualisationContext(
        algorithm="x", sample_names=None, show_sample_names=False,
        detection_box=None,
    )
    with pytest.raises(ValueError, match="detection_box"):
        vis.visualise(attributions, inputs, context=ctx)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/raitap/transparency/tests/test_detection_image_visualiser.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Create the visualiser**

Create `src/raitap/transparency/visualisers/detection_image_visualiser.py`:

```python
"""Detection image visualiser — one figure per detected box.

Renders the original image with the reference box outlined and the per-pixel
attribution heatmap overlaid. Compatible with all attribution method
families that produce per-pixel maps (gradient / perturbation / shapley /
cam / model_agnostic / surrogate). Issue #146 Phase 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from raitap.transparency.contracts import (
    ExplanationOutputSpace,
    ExplanationPayloadKind,
    ExplanationScope,
    MethodFamily,
    VisualisationContext,
)
from raitap.transparency.visualisers.base_visualiser import BaseVisualiser
from raitap.transparency.visualisers.registration import register_transparency_visualiser
from raitap.types import TaskKind

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@register_transparency_visualiser(registry_name="DetectionImage", extra="transparency")
class DetectionImageVisualiser(BaseVisualiser):
    """Render one fig per box: original image + bbox rectangle + heatmap."""

    supported_payload_kinds: ClassVar[frozenset[ExplanationPayloadKind]] = frozenset(
        {ExplanationPayloadKind.ATTRIBUTIONS}
    )
    supported_output_spaces: ClassVar[frozenset[ExplanationOutputSpace]] = frozenset(
        {ExplanationOutputSpace.DETECTION_BOXES}
    )
    supported_scopes: ClassVar[frozenset[ExplanationScope]] = frozenset(
        {ExplanationScope.LOCAL}
    )
    supported_method_families: ClassVar[frozenset[MethodFamily]] = frozenset(
        {
            MethodFamily.GRADIENT,
            MethodFamily.PERTURBATION,
            MethodFamily.SHAPLEY,
            MethodFamily.CAM,
            MethodFamily.MODEL_AGNOSTIC,
            MethodFamily.SURROGATE,
        }
    )
    supported_tasks: ClassVar[frozenset[TaskKind]] = frozenset({TaskKind.detection})
    embeds_original_input: ClassVar[bool] = True

    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        *,
        context: VisualisationContext | None = None,
        **kwargs: Any,
    ) -> Figure:
        if context is None or context.detection_box is None:
            raise ValueError(
                "DetectionImageVisualiser requires VisualisationContext.detection_box "
                "to be set; got None. This usually means the pipeline detection phase "
                "did not attach the per-box metadata to the ExplanationResult."
            )
        if inputs is None:
            raise ValueError("DetectionImageVisualiser requires inputs (original images).")

        box = context.detection_box

        # Detach + move to CPU + drop batch dim.
        img = inputs.detach().cpu()
        attr = attributions.detach().cpu()
        if img.ndim == 4:
            img = img[0]
        if attr.ndim == 4:
            attr = attr[0]
        if img.shape[0] == 3:
            img_hwc = img.permute(1, 2, 0).numpy()
        elif img.shape[0] == 1:
            img_hwc = img[0].numpy()
        else:
            img_hwc = img.permute(1, 2, 0).numpy()
        img_hwc = np.clip(img_hwc, 0.0, 1.0) if img_hwc.dtype != np.uint8 else img_hwc

        attr_2d = attr.abs().sum(dim=0).numpy() if attr.ndim == 3 else attr.numpy()
        attr_max = float(np.max(np.abs(attr_2d)))
        if attr_max > 0:
            attr_2d = attr_2d / attr_max

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_hwc, interpolation="nearest")
        ax.imshow(attr_2d, cmap="seismic", alpha=0.45, vmin=-1.0, vmax=1.0)

        x1, y1, x2, y2 = box.xyxy
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        ax.add_patch(rect)

        label_str = box.label_name if box.label_name else f"class {box.label_index}"
        ax.set_title(
            f"{label_str}: {box.score:.2f}    "
            f"[box {box.display_index} (raw {box.raw_index})]"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, img_hwc.shape[1])
        ax.set_ylim(img_hwc.shape[0], 0)    # invert Y so image coords match (origin top-left)

        fig.tight_layout()
        return fig
```

Then update `src/raitap/transparency/visualisers/__init__.py` — add the re-export alongside the existing visualiser exports:

```python
from raitap.transparency.visualisers.detection_image_visualiser import DetectionImageVisualiser

__all__ = [
    # ... existing names ...
    "DetectionImageVisualiser",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/raitap/transparency/tests/test_detection_image_visualiser.py -v`
Expected: 5 pass.

- [ ] **Step 5: Transparency regression sweep**

Run: `uv run pytest src/raitap/transparency -x -q`
Expected: all pass.

- [ ] **Step 6: Pyright**

Run: `uv run pyright src/raitap/transparency/visualisers/detection_image_visualiser.py src/raitap/transparency/tests/test_detection_image_visualiser.py`
Expected: 0 errors.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(transparency): add DetectionImageVisualiser

Single-panel figure per box: original image + reference bbox outlined +
per-pixel attribution heatmap overlay. Title carries label name (falls
back to class id) and score plus the display/raw index pair for
provenance. Declares supported_output_spaces={DETECTION_BOXES} and
supported_tasks={detection} so factory pre-flight gates it to detection
runs. Issue #146 Phase 3."
```

---

### Task 12: Docs — visualiser entry, configuration knobs, metrics one-liner

**Files:**
- Modify: `docs/modules/transparency/visualisers.md`
- Modify: `docs/modules/transparency/configuration.md`
- Modify: `docs/modules/metrics/index.md`

- [ ] **Step 1: Add the visualiser entry**

Append to `docs/modules/transparency/visualisers.md`:

```markdown
## DetectionImage

`DetectionImageVisualiser` renders one figure per detected box for any
backend whose `task_kind == detection`. The figure shows the original
image with the reference box outlined and the per-pixel attribution
heatmap overlaid; the title carries the label and score plus the box's
display/raw indices.

Compatible with all attribution method families that produce per-pixel
maps (gradient / perturbation / shapley / cam / model-agnostic /
surrogate).

```yaml
transparency:
  my_ig_explainer:
    _target_: CaptumExplainer
    algorithm: IntegratedGradients
    call:
      target: 0                  # required — wrapper exposes one scalar channel
    raitap:
      detection:
        score_threshold: 0.5     # default; drop detections below this
        max_boxes: 5             # default; cap K per sample
        iou_threshold: 0.5       # default; used by reference_match target
    visualisers:
      - _target_: DetectionImage
```

The pipeline emits one `ExplanationResult` per detected box (top-K after
threshold filtering), each carrying a `DetectionBox` with the reference
xyxy/score/label. Results from the same sample share `original_sample_index`
so the report groups them visually via the sample_id chip.
```

- [ ] **Step 2: Add the detection knobs section**

Append to `docs/modules/transparency/configuration.md`:

```markdown
## Detection knobs

For backends whose `task_kind == detection` (e.g. torchvision Faster R-CNN
/ RetinaNet / SSD), the pipeline switches to a per-box explanation loop.
Knobs live under the explainer's `raitap.detection` block:

| Key | Default | Meaning |
|---|---|---|
| `score_threshold` | `0.5` | Drop detections whose score is strictly below this before selecting boxes |
| `max_boxes` | `5` | Cap K per sample after threshold filtering, by score descending |
| `iou_threshold` | `0.5` | IoU threshold used by the `reference_match` target — perturbed predictions need at least this IoU with the original box to count |

The explainer's `call.target` MUST be `0` for detection runs (the
`ScalarDetectionWrapper` exposes one scalar channel; `target=0` selects
it). `call.target: auto_pred` is rejected with a `RaitapError` because
argmax over a 1-channel output always returns 0, masking real config
errors.
```

- [ ] **Step 3: Add the metrics one-liner**

Find the metrics section in `docs/modules/metrics/index.md` showing the
classification adapter — append the detection example:

```markdown
### Detection — torchvision-style `MeanAveragePrecision`

```yaml
metrics:
  _target_: DetectionMetrics
```

Requires the `metrics` extra. Consumes the per-sample `list[dict]`
predictions produced by detection backends + `list[dict]` targets loaded
via `data.labels.kind: detection`.
```

- [ ] **Step 4: Verify docs build**

Run: `uv run --extra docs sphinx-build -W -b html docs docs/_build/html 2>&1 | tail -20`
(If the project uses `mkdocs` instead, run the equivalent. Check
`docs/conf.py` and `pyproject.toml`.)
Expected: build succeeds without warnings.

- [ ] **Step 5: Commit**

```bash
git add docs/modules/transparency/visualisers.md docs/modules/transparency/configuration.md docs/modules/metrics/index.md
git commit -m "docs: detection visualiser + knobs + metrics one-liner

New DetectionImage entry under transparency visualisers. Detection knobs
section under transparency configuration with score_threshold /
max_boxes / iou_threshold defaults and the call.target=0 +
auto_pred-rejection note. One-liner under metrics showing
metrics: { _target_: DetectionMetrics } as the canonical detection
config. Issue #146 Phase 3 (D20)."
```

---

### Task 13: Final verification — full pytest + pyright + ruff + commit log

**Files:** (verification only — no edits unless something is broken)

- [ ] **Step 1: Full pytest sweep**

Run: `uv run pytest src/raitap -x -q`
Expected: all pass. If anything regressed, fix it on this branch in a new commit, do not skip.

- [ ] **Step 2: Pyright on all changed files**

Run:
```
uv run pyright \
  src/raitap/models/task_wrappers.py \
  src/raitap/transparency/contracts.py \
  src/raitap/transparency/results.py \
  src/raitap/transparency/semantics.py \
  src/raitap/transparency/factory.py \
  src/raitap/transparency/explainers/base_explainer.py \
  src/raitap/transparency/visualisers/detection_image_visualiser.py \
  src/raitap/transparency/visualisers/__init__.py \
  src/raitap/pipeline/outputs.py \
  src/raitap/pipeline/phases/forward_pass.py \
  src/raitap/pipeline/phases/prediction_summaries.py \
  src/raitap/pipeline/phases/evaluate_metrics.py \
  src/raitap/pipeline/phases/assess_robustness.py \
  src/raitap/pipeline/phases/assess_transparency.py \
  src/raitap/pipeline/phases/explain_detection.py \
  src/raitap/reporting/builder.py \
  src/raitap/data/data.py \
  src/raitap/configs/schema.py
```
Expected: 0 errors on the listed files. Pre-existing errors elsewhere are out of scope.

- [ ] **Step 3: Format**

Run: `uv run ruff format src/raitap`
Expected: 0 reformats. If ruff touches anything, amend the relevant commit.

- [ ] **Step 4: Confirm commits**

Run: `git log --oneline main..HEAD`
Expected: 12 commits — one per Task 1-12.

Run: `git diff --stat main..HEAD`
Expected: all changes inside `src/raitap/` and `docs/`. No `.venv/`, no `.claude/worktrees/`.

- [ ] **Step 5: Mark the plan complete**

This plan delivers Phase 3 (option C: explanations + visualiser + minimal metrics plumbing; detection metric reporting deferred).

Follow-up plans:
- Phase 3.5: detection metric reporting (mAP table, class-AP table in PDF/HTML).
- Phase 4: `DetectionAdversarialLoss` (DAG-style) + foolbox / torchattacks wiring.
- Phase 5: `configs/model/fasterrcnn.yaml`, detection dataset config, ONNX export validation, full E2E test, docs page on adding a new task family.

When all 12 commits land + the full test sweep is green: open the PR
referencing `docs/superpowers/specs/2026-05-17-detection-image-visualiser-design.md`
as the design source.
