# Detection Task-Family Foundation Implementation Plan

> **2026-05-17 v2 port note.** This plan was originally written against the pre-#167 architecture (`SemanticallyDescribable` ABC + `register=False` opt-out in `src/raitap/semantics_base.py`). Between v1 PR #151 opening and v2 landing, `main` refactored adapter registration to family decorators (`@register_transparency_adapter` etc.) backed by `AdapterMixin` in `src/raitap/_adapters.py`. The TDD structure and code shapes below are still accurate up to these symbol substitutions:
>
> | v1 plan reference | v2 reality (on `main`) |
> |-------------------|------------------------|
> | `src/raitap/semantics_base.py` (TaskKind home) | `src/raitap/types.py` |
> | `class SemanticallyDescribable` | `class AdapterMixin` (`src/raitap/_adapters.py:95`) |
> | `TaskKind.CLASSIFICATION` etc. (uppercase) | `TaskKind.classification` etc. (lowercase, matches OmegaConf convention used by sibling `Hardware` / `Task` enums) |
> | `__init_subclass__` validator on `SemanticallyDescribable` | No validator — `__init_subclass__` chain was removed by PR #167; `supported_tasks` is a plain ClassVar default |
> | `register=False` opt-out | Not applicable (no validator) |
> | `AbstractExplainer` | `BaseExplainer` (`src/raitap/transparency/explainers/base_explainer.py:36`) |
>
> The v2 commits on branch `146-detection-task-family-v2` are `932253c` (TaskKind + supported_tasks), `3202ed6` (task_kind on backend), `47006a3` (DETECTION branch in semantics) plus cherry-picks `ee63428..ba97693` for the survivable pieces. Original v1 PR #151 is superseded.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the cross-cutting taxonomy and scalar-wrapper foundation from issue #146 so future detection / segmentation / seq2seq adapters can plug into the existing transparency + robustness pipelines unchanged.

**Architecture:**
1. New `TaskKind` enum in `raitap.semantics_base` (shared, neutral location used by both transparency and robustness).
2. Extend `ExplanationOutputSpace` with detection / segmentation / bbox-regression members.
3. Add `supported_tasks: ClassVar[frozenset[TaskKind]]` on `SemanticallyDescribable`, defaulting to `{CLASSIFICATION}` so existing adapters keep current behaviour without edits.
4. Add `task_kind` property to `ModelBackend`. `TorchBackend` infers DETECTION from torchvision detection classes; everything else stays CLASSIFICATION.
5. Add `DetectionTarget` (`class_score` / `objectness` / `bbox_l2`) + a `ScalarDetectionWrapper(nn.Module)` that lets existing scalar-output explainers and attacks run against a Faster R-CNN unchanged.
6. Extend `infer_output_space` with a DETECTION branch so the transparency pipeline tags detection explanations correctly.

**Scope (this plan):** Phase 1 + Phase 2 of issue #146. Visualiser (Phase 3), adversarial loss (Phase 4), fasterrcnn config + E2E + docs (Phase 5) get follow-up plans.

**Tech Stack:** Python 3.12, PyTorch, torchvision, pytest, `uv run`.

---

## File Structure

**Create:**
- `src/raitap/models/task_wrappers.py` — `DetectionTarget` + `ScalarDetectionWrapper`.
- `src/raitap/models/tests/test_task_wrappers.py` — unit tests for both.
- `src/raitap/tests/test_task_kind.py` — taxonomy tests.

**Modify:**
- `src/raitap/semantics_base.py` — add `TaskKind` enum + `supported_tasks` ClassVar (default `{CLASSIFICATION}`).
- `src/raitap/transparency/contracts.py` — extend `ExplanationOutputSpace`.
- `src/raitap/transparency/semantics.py` — `infer_output_space` DETECTION branch.
- `src/raitap/models/backend.py` — `task_kind` property on `ModelBackend` / `TorchBackend` / `OnnxBackend`.

**Test:**
- `src/raitap/models/tests/test_backend.py` — `task_kind` cases.
- `src/raitap/transparency/tests/test_semantics.py` — DETECTION branch.

---

## Conventions

- All tests run via `uv run pytest <path> -v`.
- Format: `uv run ruff format .` after each edit batch.
- Commits: Conventional Commits (`feat:`, `test:`, `refactor:`). Body explains *why* when non-obvious.
- Each Task ends with a single commit covering the task's tests + impl. Mark task complete only after the commit lands and `uv run pytest src/raitap` passes for touched modules.

---

### Task 1: Add `TaskKind` enum + `supported_tasks` ClassVar

**Files:**
- Modify: `src/raitap/semantics_base.py`
- Create: `src/raitap/tests/test_task_kind.py`

- [ ] **Step 1: Write the failing test**

Create `src/raitap/tests/test_task_kind.py`:

```python
"""Tests for the shared TaskKind taxonomy and supported_tasks ClassVar."""

from __future__ import annotations

import pytest

from raitap.semantics_base import SemanticallyDescribable, TaskKind


def test_task_kind_members():
    assert {member.value for member in TaskKind} == {
        "classification",
        "detection",
        "segmentation",
        "seq2seq",
        "regression",
    }


def test_supported_tasks_defaults_to_classification():
    class _Adapter(SemanticallyDescribable[str]):
        algorithm_registry = {"foo": "bar"}

    assert _Adapter.supported_tasks == frozenset({TaskKind.CLASSIFICATION})


def test_supported_tasks_can_be_overridden():
    class _DetectionAdapter(SemanticallyDescribable[str]):
        algorithm_registry = {"foo": "bar"}
        supported_tasks = frozenset({TaskKind.DETECTION})

    assert _DetectionAdapter.supported_tasks == frozenset({TaskKind.DETECTION})


def test_supported_tasks_must_be_frozenset_of_task_kind():
    with pytest.raises(TypeError, match="supported_tasks"):

        class _BadAdapter(SemanticallyDescribable[str]):
            algorithm_registry = {"foo": "bar"}
            supported_tasks = frozenset({"detection"})  # type: ignore[assignment]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/raitap/tests/test_task_kind.py -v`
Expected: All 4 tests FAIL with `ImportError: cannot import name 'TaskKind'`.

- [ ] **Step 3: Add `TaskKind` + `supported_tasks` to `semantics_base.py`**

Edit `src/raitap/semantics_base.py`. Replace the existing imports + `SemanticallyDescribable` class body to add the enum and the new ClassVar with validation:

```python
"""Cross-module base interface for adapters that publish an algorithm registry.

Both ``raitap.transparency`` and ``raitap.robustness`` describe their adapters
by mapping algorithm-name → typed semantics payload (``frozenset[MethodFamily]``
for explainers, ``AssessorSemanticsHints`` for assessors). This module provides
the single shared contract:

* ``SemanticallyDescribable[T]`` — generic ABC; subclasses declare
  ``algorithm_registry: ClassVar[Mapping[str, T]]`` as a non-empty mapping.
* ``__init_subclass__`` enforces the registry at class-definition time so
  configuration mistakes fail at import, not at runtime mid-pipeline.
* ``TaskKind`` — task-family taxonomy (classification / detection / segmentation
  / seq2seq / regression) shared across the transparency and robustness modules.
* ``supported_tasks`` ClassVar on every adapter, defaulting to
  ``{TaskKind.CLASSIFICATION}`` so legacy adapters keep current behaviour
  without edits.

Intermediate abstract base classes (e.g. ``BaseAssessor``,
``EmpiricalAttackAssessor``, ``AbstractExplainer``) opt out of validation by
passing ``register=False`` in their class signature; concrete adapters always
opt in (default).
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, ClassVar, Generic, TypeVar

T = TypeVar("T")


class TaskKind(StrEnum):
    """Model task family.

    Adapters declare which task families they support via
    ``supported_tasks: ClassVar[frozenset[TaskKind]]``. The default is
    ``{CLASSIFICATION}`` so existing adapters stay correct without explicit
    declaration.
    """

    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    SEQ2SEQ = "seq2seq"
    REGRESSION = "regression"


class SemanticallyDescribable(ABC, Generic[T]):
    """Adapter that publishes an algorithm-name → hints registry as a ClassVar.

    Subclasses must declare a non-empty ``algorithm_registry`` ClassVar at
    class-definition time. Pass ``register=False`` on the class line for
    abstract intermediate classes that don't (yet) carry concrete algorithms.

    The ``Generic[T]`` parameter is enforced statically only — pyright requires
    a ``# type: ignore[misc]`` on the base ClassVar declaration because PEP
    526 forbids ``ClassVar`` parameterised by a TypeVar. Concrete subclasses
    annotate ``algorithm_registry`` with the resolved type and pyright checks
    them normally.
    """

    algorithm_registry: ClassVar[Mapping[str, Any]]  # type: ignore[misc]
    """Concrete subclasses narrow the value type to ``Mapping[str, T]``."""

    supported_tasks: ClassVar[frozenset[TaskKind]] = frozenset({TaskKind.CLASSIFICATION})
    """Task families this adapter supports. Defaults to ``{CLASSIFICATION}``."""

    def __init_subclass__(cls, *, register: bool = True, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        raw_tasks = cls.__dict__.get("supported_tasks")
        if raw_tasks is not None:
            if not isinstance(raw_tasks, frozenset) or not all(
                isinstance(item, TaskKind) for item in raw_tasks
            ):
                raise TypeError(
                    f"{cls.__name__}.supported_tasks must be a "
                    "frozenset[TaskKind]."
                )
            if not raw_tasks:
                raise TypeError(
                    f"{cls.__name__}.supported_tasks must contain at least "
                    "one TaskKind member."
                )
        if not register:
            return
        registry = cls.__dict__.get("algorithm_registry")
        if not isinstance(registry, Mapping) or not registry:
            raise TypeError(
                f"{cls.__name__} must declare a non-empty "
                "``algorithm_registry: ClassVar[Mapping[str, ...]]`` ClassVar. "
                "Abstract intermediate classes can opt out via "
                "``class Foo(SemanticallyDescribable, register=False): ...``."
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/raitap/tests/test_task_kind.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run the full transparency + robustness test sweep so existing adapters still load**

Run: `uv run pytest src/raitap/transparency src/raitap/robustness src/raitap/tests -x -q`
Expected: all pass. Failure means a real adapter declared `supported_tasks` incorrectly somewhere — fix that adapter, do not relax the validator.

- [ ] **Step 6: Commit**

```bash
git add src/raitap/semantics_base.py src/raitap/tests/test_task_kind.py
git commit -m "$(cat <<'EOF'
feat(semantics): add TaskKind enum + supported_tasks ClassVar

Issue #146 groundwork. New shared taxonomy lets transparency / robustness
adapters declare which task families they accept (classification by default
so existing adapters keep current behaviour without edits).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Extend `ExplanationOutputSpace` with detection / segmentation members

**Files:**
- Modify: `src/raitap/transparency/contracts.py:49-56`
- Test: `src/raitap/transparency/tests/test_contracts.py`

- [ ] **Step 1: Write the failing test**

Create or append to `src/raitap/transparency/tests/test_contracts.py`:

```python
from raitap.transparency.contracts import ExplanationOutputSpace


def test_explanation_output_space_includes_detection_members():
    values = {member.value for member in ExplanationOutputSpace}
    assert "detection_boxes" in values
    assert "segmentation_mask" in values
    assert "bbox_regression" in values
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/raitap/transparency/tests/test_contracts.py::test_explanation_output_space_includes_detection_members -v`
Expected: FAIL (one or more values missing).

- [ ] **Step 3: Extend the enum**

Edit `src/raitap/transparency/contracts.py:49-56`. Replace the `ExplanationOutputSpace` class with:

```python
class ExplanationOutputSpace(StrEnum):
    """Coordinate space represented by attribution values."""

    INPUT_FEATURES = "input_features"
    INTERPRETABLE_FEATURES = "interpretable_features"
    LAYER_ACTIVATION = "layer_activation"
    IMAGE_SPATIAL_MAP = "image_spatial_map"
    TOKEN_SEQUENCE = "token_sequence"
    DETECTION_BOXES = "detection_boxes"
    SEGMENTATION_MASK = "segmentation_mask"
    BBOX_REGRESSION = "bbox_regression"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/raitap/transparency/tests/test_contracts.py -v`
Expected: pass.

- [ ] **Step 5: Run the transparency suite — new members are additive but verify nothing was importing the enum strictly**

Run: `uv run pytest src/raitap/transparency -x -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/raitap/transparency/contracts.py src/raitap/transparency/tests/test_contracts.py
git commit -m "$(cat <<'EOF'
feat(transparency): add detection / segmentation / bbox_regression output spaces

Issue #146 groundwork. Future DetectionImageVisualiser /
SegmentationVisualiser tag their outputs with these spaces; existing
visualisers untouched.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Add `task_kind` property to `ModelBackend`

**Files:**
- Modify: `src/raitap/models/backend.py`
- Test: `src/raitap/models/tests/test_backend.py`

- [ ] **Step 1: Write the failing tests**

Append to `src/raitap/models/tests/test_backend.py` (or create if absent):

```python
from __future__ import annotations

import torch
from torch import nn

from raitap.models.backend import TorchBackend
from raitap.semantics_base import TaskKind


class _Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_torch_backend_defaults_to_classification():
    backend = TorchBackend(_Linear())
    assert backend.task_kind is TaskKind.CLASSIFICATION


def test_torch_backend_detects_torchvision_detection_model():
    pytest = __import__("pytest")
    try:
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn_v2,  # noqa: F401
        )
    except ImportError:
        pytest.skip("torchvision detection models unavailable")

    model = fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=2)
    backend = TorchBackend(model)
    assert backend.task_kind is TaskKind.DETECTION


def test_torch_backend_task_kind_can_be_overridden_in_constructor():
    backend = TorchBackend(_Linear(), task_kind=TaskKind.REGRESSION)
    assert backend.task_kind is TaskKind.REGRESSION
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/raitap/models/tests/test_backend.py -v -k task_kind`
Expected: all three FAIL — `AttributeError: 'TorchBackend' object has no attribute 'task_kind'` (or `TypeError` for the constructor-override test).

- [ ] **Step 3: Add `task_kind` to `ModelBackend` and override on `TorchBackend`**

Edit `src/raitap/models/backend.py`:

1. Add a new import after the existing `torch` import block (top of file):

```python
from raitap.semantics_base import TaskKind
```

2. Add a new helper after `_NUMPY_DTYPES_BY_ONNX_TYPE`:

```python
def _is_torchvision_detection_model(model: nn.Module) -> bool:
    """Return True for any torchvision detection model (Faster R-CNN, RetinaNet, etc.)."""
    try:
        from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
        from torchvision.models.detection.retinanet import RetinaNet
        from torchvision.models.detection.ssd import SSD
    except ImportError:
        return False
    return isinstance(model, GeneralizedRCNN | RetinaNet | SSD)
```

3. Add a new abstract property on `ModelBackend` (insert after the existing `hardware_label` property, around line 43):

```python
    @property
    def task_kind(self) -> TaskKind:
        """Task family this backend serves. Defaults to ``CLASSIFICATION``."""
        return TaskKind.CLASSIFICATION
```

4. Edit `TorchBackend.__init__` (`backend.py:67-69`) and add a `task_kind` property:

```python
    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device | None = None,
        task_kind: TaskKind | None = None,
    ) -> None:
        self.model = model
        self.device = torch.device("cpu") if device is None else device
        self._task_kind = task_kind if task_kind is not None else self._infer_task_kind(model)

    @staticmethod
    def _infer_task_kind(model: nn.Module) -> TaskKind:
        if _is_torchvision_detection_model(model):
            return TaskKind.DETECTION
        return TaskKind.CLASSIFICATION

    @property
    def task_kind(self) -> TaskKind:
        return self._task_kind
```

(Leave `_prepare_inputs`, `_prepare_kwargs`, `__call__`, `as_model_for_explanation` untouched in this task.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/raitap/models/tests/test_backend.py -v -k task_kind`
Expected: 3 passed.

- [ ] **Step 5: Run the model suite + a smoke of pipeline tests**

Run: `uv run pytest src/raitap/models src/raitap/tests/test_run_main.py -x -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/raitap/models/backend.py src/raitap/models/tests/test_backend.py
git commit -m "$(cat <<'EOF'
feat(models): add task_kind property on ModelBackend

TorchBackend auto-detects torchvision detection models (Faster R-CNN,
RetinaNet, SSD) and reports TaskKind.DETECTION. All other backends and
models stay CLASSIFICATION. Issue #146 groundwork.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `DetectionTarget` — reduce list-of-dicts to a scalar

**Files:**
- Create: `src/raitap/models/task_wrappers.py`
- Create: `src/raitap/models/tests/test_task_wrappers.py`

- [ ] **Step 1: Write the failing tests**

Create `src/raitap/models/tests/test_task_wrappers.py`:

```python
"""Tests for DetectionTarget — scalar reducer for detection model outputs."""

from __future__ import annotations

import torch

from raitap.models.task_wrappers import DetectionTarget


def _sample_detection_output() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 12.0, 12.0]]),
            "scores": torch.tensor([0.9, 0.4]),
            "labels": torch.tensor([1, 2]),
        },
    ]


def test_class_score_returns_score_at_box_index():
    target = DetectionTarget(box_idx=0, mode="class_score")
    value = target(_sample_detection_output())
    assert torch.allclose(value, torch.tensor(0.9))


def test_class_score_returns_zero_for_missing_box():
    target = DetectionTarget(box_idx=99, mode="class_score")
    value = target(_sample_detection_output())
    assert torch.allclose(value, torch.tensor(0.0))


def test_objectness_sums_scores_in_batch():
    target = DetectionTarget(box_idx=0, mode="objectness")
    value = target(_sample_detection_output())
    assert torch.allclose(value, torch.tensor(0.9 + 0.4))


def test_bbox_l2_returns_squared_norm_of_first_box():
    target = DetectionTarget(box_idx=0, mode="bbox_l2")
    value = target(_sample_detection_output())
    expected = torch.tensor(0.0**2 + 0.0**2 + 10.0**2 + 10.0**2)
    assert torch.allclose(value, expected)


def test_rejects_unknown_mode():
    pytest = __import__("pytest")
    with pytest.raises(ValueError, match="mode"):
        DetectionTarget(box_idx=0, mode="nonsense")  # type: ignore[arg-type]


def test_rejects_non_list_output():
    target = DetectionTarget(box_idx=0, mode="class_score")
    pytest = __import__("pytest")
    with pytest.raises(TypeError):
        target(torch.zeros(3))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/raitap/models/tests/test_task_wrappers.py -v`
Expected: 6 tests FAIL with `ImportError: cannot import name 'DetectionTarget'`.

- [ ] **Step 3: Create `DetectionTarget`**

Create `src/raitap/models/task_wrappers.py`:

```python
"""Task-generic helpers that map structured model outputs to a scalar.

Detection models (e.g. ``fasterrcnn_resnet50_fpn_v2``) return
``list[dict[str, Tensor]]``. The existing transparency explainers and
robustness adversarial attacks assume a scalar-per-sample output. This
module bridges that gap with two pieces:

* :class:`DetectionTarget` — reduces a detection model's output to a
  single ``torch.Tensor`` scalar via one of three modes
  (``class_score`` / ``objectness`` / ``bbox_l2``).
* :class:`ScalarDetectionWrapper` — an ``nn.Module`` that wraps a
  detection model and applies a ``DetectionTarget`` so existing
  scalar-output adapters (Captum, SHAP, torchattacks, foolbox) can
  consume detection models unchanged.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

DetectionTargetMode = Literal["class_score", "objectness", "bbox_l2"]
_VALID_MODES: frozenset[str] = frozenset({"class_score", "objectness", "bbox_l2"})


class DetectionTarget:
    """Reduce a torchvision-style detection output to a scalar tensor.

    Parameters
    ----------
    box_idx:
        Box index inside each sample's prediction dict. Out-of-range
        indices return ``0.0`` so explainers don't have to special-case
        empty predictions.
    mode:
        ``"class_score"`` — score of the box at ``box_idx`` summed over
        the batch (one scalar per call).
        ``"objectness"`` — sum of all box scores across the batch.
        ``"bbox_l2"`` — squared L2 norm of the first sample's
        ``box_idx``-th bounding box coordinates.
    """

    def __init__(self, box_idx: int, mode: DetectionTargetMode) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"DetectionTarget mode must be one of {sorted(_VALID_MODES)}; got {mode!r}."
            )
        self.box_idx = int(box_idx)
        self.mode = mode

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
            return torch.stack(
                [item["scores"].sum() if item["scores"].numel() > 0 else torch.tensor(0.0)
                 for item in model_out]
            ).sum()

        if self.mode == "class_score":
            per_sample = []
            for item in model_out:
                scores = item.get("scores")
                if scores is None or scores.numel() <= self.box_idx:
                    per_sample.append(torch.tensor(0.0, device=scores.device if scores is not None else None))
                else:
                    per_sample.append(scores[self.box_idx])
            return torch.stack(per_sample).sum()

        # mode == "bbox_l2"
        boxes = model_out[0].get("boxes")
        if boxes is None or boxes.numel() == 0 or boxes.shape[0] <= self.box_idx:
            return torch.tensor(0.0)
        return (boxes[self.box_idx] ** 2).sum()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/raitap/models/tests/test_task_wrappers.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/raitap/models/task_wrappers.py src/raitap/models/tests/test_task_wrappers.py
git commit -m "$(cat <<'EOF'
feat(models): add DetectionTarget scalar reducer

Maps torchvision-style list[dict] detection outputs to a single scalar
(class_score / objectness / bbox_l2) so existing scalar-output explainers
and adversarial attacks plug into detection models unchanged. Issue #146
Phase 2 groundwork.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `ScalarDetectionWrapper(nn.Module)`

**Files:**
- Modify: `src/raitap/models/task_wrappers.py`
- Modify: `src/raitap/models/tests/test_task_wrappers.py`

- [ ] **Step 1: Write the failing tests**

Append to `src/raitap/models/tests/test_task_wrappers.py`:

```python
from raitap.models.task_wrappers import ScalarDetectionWrapper


class _FakeDetector(nn.Module):
    def forward(self, images):
        batch_size = images.shape[0]
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                "scores": torch.tensor([0.7]),
                "labels": torch.tensor([5]),
            }
            for _ in range(batch_size)
        ]


def test_scalar_detection_wrapper_returns_batched_logit_tensor():
    wrapper = ScalarDetectionWrapper(
        _FakeDetector(), target=DetectionTarget(box_idx=0, mode="class_score")
    )
    images = torch.zeros(2, 3, 8, 8)
    out = wrapper(images)
    # Shape contract: (batch, 1) so existing classification-shaped explainers
    # (which call ``output[:, target_class]``) work unchanged.
    assert out.shape == (2, 1)
    assert torch.allclose(out, torch.tensor([[0.7], [0.7]]))


def test_scalar_detection_wrapper_is_an_nn_module():
    wrapper = ScalarDetectionWrapper(
        _FakeDetector(), target=DetectionTarget(box_idx=0, mode="objectness")
    )
    assert isinstance(wrapper, nn.Module)


def test_scalar_detection_wrapper_eval_propagates_to_inner_model():
    detector = _FakeDetector()
    detector.train()
    wrapper = ScalarDetectionWrapper(
        detector, target=DetectionTarget(box_idx=0, mode="class_score")
    )
    wrapper.eval()
    assert not detector.training
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/raitap/models/tests/test_task_wrappers.py -v -k scalar_detection`
Expected: FAIL with `ImportError: cannot import name 'ScalarDetectionWrapper'`.

- [ ] **Step 3: Add `ScalarDetectionWrapper` below `DetectionTarget` in `task_wrappers.py`**

Append to `src/raitap/models/task_wrappers.py`:

```python
class ScalarDetectionWrapper(nn.Module):
    """Make a detection model look like a scalar-output classification model.

    Existing explainers (Captum / SHAP / Grad-CAM) and gradient-based attacks
    (torchattacks / foolbox) call ``model(x)[:, target_class]`` and
    differentiate the result. This wrapper takes any module whose forward
    returns ``list[dict[str, Tensor]]`` and reduces each sample's prediction
    to a single scalar via :class:`DetectionTarget`, returning a tensor of
    shape ``(batch, 1)``.
    """

    def __init__(self, model: nn.Module, *, target: DetectionTarget) -> None:
        super().__init__()
        self.model = model
        self.target = target

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.model(inputs)
        if not isinstance(outputs, list):
            raise TypeError(
                "ScalarDetectionWrapper expected list[dict] from the wrapped "
                f"detection model; got {type(outputs).__name__}."
            )
        per_sample: list[torch.Tensor] = []
        for sample in outputs:
            scalar = self.target([sample])
            per_sample.append(scalar.reshape(()))
        if not per_sample:
            return torch.zeros((0, 1), device=inputs.device)
        return torch.stack(per_sample).reshape(-1, 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/raitap/models/tests/test_task_wrappers.py -v`
Expected: 9 passed (6 from Task 4 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/raitap/models/task_wrappers.py src/raitap/models/tests/test_task_wrappers.py
git commit -m "$(cat <<'EOF'
feat(models): add ScalarDetectionWrapper nn.Module

Wraps a detection model + DetectionTarget into a scalar-output nn.Module
that returns (batch, 1) tensors. Existing classification-shaped explainers
and gradient-based attacks consume it unchanged. Issue #146 Phase 2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `infer_output_space` DETECTION branch

**Files:**
- Modify: `src/raitap/transparency/semantics.py:147-231`
- Modify: `src/raitap/transparency/tests/test_semantics.py`

- [ ] **Step 1: Write the failing test**

Append to `src/raitap/transparency/tests/test_semantics.py` (create if absent):

```python
from __future__ import annotations

import torch

from raitap.semantics_base import TaskKind
from raitap.transparency.contracts import (
    ExplanationOutputSpace,
    InputKind,
    InputSpec,
    TensorLayout,
)
from raitap.transparency.semantics import infer_output_space


def test_infer_output_space_returns_detection_boxes_for_detection_task():
    input_spec = InputSpec(
        kind=InputKind.IMAGE,
        shape=(1, 3, 224, 224),
        layout=TensorLayout.BATCH_CHANNEL_HEIGHT_WIDTH,
    )
    attrs = torch.zeros(1, 3, 224, 224)
    result = infer_output_space(
        input_spec=input_spec,
        attributions=attrs,
        task_kind=TaskKind.DETECTION,
    )
    assert result.space is ExplanationOutputSpace.DETECTION_BOXES


def test_infer_output_space_classification_unchanged_when_task_kind_omitted():
    input_spec = InputSpec(
        kind=InputKind.IMAGE,
        shape=(1, 3, 224, 224),
        layout=TensorLayout.BATCH_CHANNEL_HEIGHT_WIDTH,
    )
    attrs = torch.zeros(1, 3, 224, 224)
    # No task_kind → default classification path → INPUT_FEATURES (existing
    # behaviour, must not regress).
    result = infer_output_space(input_spec=input_spec, attributions=attrs)
    assert result.space is ExplanationOutputSpace.INPUT_FEATURES
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/raitap/transparency/tests/test_semantics.py -v -k detection`
Expected: FAIL — `infer_output_space` does not accept `task_kind`.

- [ ] **Step 3: Extend `infer_output_space` signature + DETECTION branch**

Edit `src/raitap/transparency/semantics.py`:

1. Add the import at the top:

```python
from raitap.semantics_base import TaskKind
```

2. Replace the `infer_output_space` signature (`semantics.py:147-156`) and add the DETECTION branch immediately after the CAM branch:

```python
def infer_output_space(
    *,
    input_spec: InputSpec,
    attributions: object | None = None,
    explainer: object | None = None,
    algorithm: str | None = None,
    method_families: frozenset[MethodFamily] | None = None,
    layer_path: str | None = None,
    feature_names: Sequence[str] | None = None,
    task_kind: TaskKind | None = None,
) -> OutputSpaceSpec:
    """Infer deterministic output-space metadata from input and method semantics."""

    if task_kind is TaskKind.DETECTION:
        shape = _shape_tuple(getattr(attributions, "shape", None))
        features = list(feature_names) if feature_names is not None else input_spec.feature_names
        return OutputSpaceSpec(
            space=ExplanationOutputSpace.DETECTION_BOXES,
            shape=shape,
            layout=input_spec.layout,
            layer_path=layer_path,
            feature_names=features,
        )

    resolved_method_families = _resolve_method_families(
        method_families=method_families,
        explainer=explainer,
        algorithm=algorithm,
    )
    # ... (rest of the function unchanged)
```

(Preserve every line below `resolved_method_families = ...` exactly as it is today; the only edits are the new kwarg, the docstring stays the same, and the early-return DETECTION block.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/raitap/transparency/tests/test_semantics.py -v`
Expected: all pass, including the legacy ones.

- [ ] **Step 5: Run the transparency suite for regressions**

Run: `uv run pytest src/raitap/transparency -x -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/raitap/transparency/semantics.py src/raitap/transparency/tests/test_semantics.py
git commit -m "$(cat <<'EOF'
feat(transparency): infer DETECTION_BOXES output space for detection task

Adds an optional task_kind kwarg to infer_output_space. When DETECTION,
the inferred space is DETECTION_BOXES; classification path is unchanged.
Issue #146 Phase 1 groundwork.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Final verification + push-up to summary

**Files:** (verification only — no edits)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -x -q`
Expected: all pass. If anything regressed, fix on this branch in a new commit, do not skip.

- [ ] **Step 2: Type check the changed files**

Run: `uv run pyright src/raitap/semantics_base.py src/raitap/transparency/contracts.py src/raitap/transparency/semantics.py src/raitap/models/backend.py src/raitap/models/task_wrappers.py`
Expected: 0 errors / 0 warnings on the listed files. Pre-existing errors elsewhere are out of scope for this plan but worth a one-line note in the PR description.

- [ ] **Step 3: Format**

Run: `uv run ruff format src/raitap/semantics_base.py src/raitap/transparency src/raitap/models`
Expected: 0 reformats (or amend the last commit if ruff touches anything).

- [ ] **Step 4: Confirm with git log + git diff stat**

Run: `git log --oneline main..HEAD`
Expected: exactly six commits (one per Task 1-6).

Run: `git diff --stat main..HEAD`
Expected: all changes inside `src/raitap/semantics_base.py`, `src/raitap/transparency/`, `src/raitap/models/`, `src/raitap/tests/`. No `.venv/`, no `.claude/worktrees/`, no docs.

- [ ] **Step 5: Mark the plan complete**

This plan delivers Phase 1 + Phase 2 of issue #146. Follow-up plans:

- `2026-05-XX-detection-image-visualiser.md` — Phase 3.
- `2026-05-XX-detection-adversarial-loss.md` — Phase 4.
- `2026-05-XX-fasterrcnn-e2e.md` — Phase 5 (model config, dataset, ONNX export, docs page).

Do not open follow-ups until this plan's six commits are merged or reviewed; later phases assume the taxonomy + scalar wrapper already exist on `main`.
