---
title: "Adding a task family"
description: "How to add a new task family (segmentation, seq2seq, ...) to RAITAP: add a TaskKind, implement a TaskFamily strategy class, register its adapter + visualiser, and declare the model's task. Detection is the worked example."
myst:
  html_meta:
    "description": "How to add a new task family (segmentation, seq2seq, ...) to RAITAP: add a TaskKind, implement a TaskFamily strategy class, register its adapter + visualiser, and declare the model's task. Detection is the worked example."
---

# Adding a task family

A **task family** is the shape of the model's forward output: classification returns a `(batch, num_classes)` tensor, detection returns `list[dict[str, Tensor]]`, segmentation a per-pixel mask, seq2seq variable-length token sequences. Each family is one strategy object: a `TaskFamily` subclass in `src/raitap/task_families/` that owns every task-specific behaviour the pipeline phases used to branch on. `ClassificationFamily` and `DetectionFamily` are the two registered families; copy whichever is closer to your case.

This page covers adding a new family. Adding a new *adapter* to an existing family is much smaller: see {doc}`adding-an-adapter`.

## When to add a task family

Trigger: the forward output is not a scalar-per-sample tensor and the user-facing semantics of explanation / metrics / robustness differ from classification.

If the output reduces to a scalar via a thin wrapper without changing what the user sees (e.g. "explain the score of box #0" instead of "explain each box"), you do not need a new family. Write a wrapper and reuse `TaskKind.classification` (pattern: `ScalarDetectionWrapper` in `src/raitap/models/task_wrappers.py`). Detection earned a new family because per-box semantics differ from per-class: K boxes per sample, each with its own attribution map and ground-truth match.

## 1. Add a `TaskKind` member

`TaskKind` in `src/raitap/types.py` is a `StrEnum`. Add one member; name and value must be identical and lowercase so OmegaConf round-trips it through YAML. This is the only edit to a shared file.

```python
class TaskKind(StrEnum):
    classification = "classification"
    detection = "detection"
    segmentation = "segmentation"
```

## 2. Implement the `TaskFamily` subclass

Add `src/raitap/task_families/<family>.py`. The `@task_family` decorator instantiates the class once and registers that singleton under its `kind`. Two class attributes plus the `TaskFamily` Protocol members (`src/raitap/task_families/base.py`):

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap.task_families.registry import task_family
from raitap.transparency.contracts import ExplanationOutputSpace
from raitap.types import TaskKind
from raitap.utils.lazy import lazy_import

if TYPE_CHECKING:
    import torch

    from raitap.task_families.base import ExplainContext, ForwardContext
else:
    torch = lazy_import("torch")  # this module is imported at CLI startup, before torch exists


@task_family
class SegmentationFamily:
    kind: TaskKind = TaskKind.segmentation
    fixed_output_space: ExplanationOutputSpace | None = ExplanationOutputSpace.IMAGE_SPATIAL_MAP

    def validate_payload(self, payload: object) -> None:
        if not isinstance(payload, torch.Tensor):
            raise ValueError("segmentation payload must be a (N, C, H, W) mask tensor.")

    def explain(self, ctx: ExplainContext) -> list:
        ...  # the rest of the Protocol members below
```

Use `lazy_import("torch")`, never a top-level `import torch`: this file is imported eagerly for `@task_family` registration, and the bare CLI bootstrap imports it before installing torch. A top-level import crashes `raitap --demo -y` (see {doc}`adding-an-adapter`).

Class attributes:

- `kind: TaskKind`: the member from step 1.
- `fixed_output_space: ExplanationOutputSpace | None`: `None` means "infer the output space dynamically" (classification: CAM vs input-features). A fixed value pins it (detection: `DETECTION_BOXES`).

Methods (signatures take `ctx` / `payload` / `tensor` / `cfg` as the Protocol defines):

- `validate_payload(payload)`: raise `ValueError` if the forward payload is the wrong type for this family.
- `adapt_loaded_inputs(tensor)`: shape the freshly-loaded dense tensor. Classification returns it as-is; detection unbinds it into a ragged `list[(C, H, W)]`.
- `validate_inputs(tensor)`: raise if the post-adapt inputs break this family's contract.
- `load_labels(cfg, *, tensor, sample_ids)`: load labels in this family's on-disk shape, or return `None` when no label source is set.
- `validate_labels(labels)`: raise when loaded labels are the wrong shape (e.g. a tensor where this family wants `list[dict]`); a mismatch means model and data declare different families.
- `extract_forward(ctx, *, batch_size)`: run the backend forward in chunks and return this family's payload.
- `explain(ctx)`: return `list[ExplanationResult]`. Classification runs the explainer once over the batch; detection runs a per-box K-loop.
- `metrics_inputs(config, forward_output, labels)`: adapt payload + labels into `(preds, targets)` for the metric adapters, or `None` to skip metrics.
- `supports_robustness()`: return whether the robustness phase runs for this family.
- `prediction_summaries(payload, *, sample_ids, targets)`: per-sample summary rows, or `None` when the concept does not apply (detection returns `None`).
- `allows_preprocessing` (property): whether data/model preprocessing transforms are allowed.
- `matches_model(model)`: *optional.* Return `True` if this family recognises `model` by architecture, for backend auto-inference (step 4). Implement it only for auto-detectable families. Classification is the fallback and returns `False`.

`ClassificationFamily` and `DetectionFamily` are the two worked examples; read both before writing yours.

## 3. Register the metric adapter and visualiser

Same decorators as any adapter ({doc}`adding-an-adapter`). The metric computer ties to your family by `registry_name` (selected via `metrics=<name>`); the visualiser ties to it by `supported_tasks`, which makes the transparency pre-flight reject an incompatible config up front instead of failing deep in the explain loop.

```python
from raitap.metrics.registration import metrics_adapter
from raitap.transparency.visualisers.registration import transparency_visualiser

@metrics_adapter(registry_name="segmentation", extra="metrics", schema=SegmentationMetricsConfig)
class SegmentationMetrics(BaseMetricComputer): ...

@transparency_visualiser(
    registry_name="segmentation_mask",
    supported_tasks={TaskKind.segmentation},                          # the pre-flight gate
    supported_output_spaces={ExplanationOutputSpace.IMAGE_SPATIAL_MAP},
    supported_payload_kinds={ExplanationPayloadKind.ATTRIBUTIONS},
)
class SegmentationMaskVisualiser(BaseVisualiser): ...
```

## 4. Declare the model's task

The pipeline reads `backend.task_kind`. Pass it explicitly, or rely on auto-inference: `TorchBackend._infer_task_kind` calls each registered family's `matches_model` and raises if more than one matches.

```python
backend = TorchBackend(model, task_kind=TaskKind.segmentation)  # explicit
backend = TorchBackend(model)                                   # auto: scans matches_model
```

Implement `matches_model` only for an architecture-detectable family (torchvision detectors are recognised this way); otherwise pass `task_kind=` explicitly. Backends have their own page: {doc}`adding-a-backend`.

## 5. Data-layer boundary

The `Data` layer loads images, tabular files, and (via the `data/inputs` parser registry plus `model.tokenizer`) text into tensors, then routes by a `task_kind` file-loading check. Image, tabular, and text classification all work within the current loader (see {doc}`adding-an-input-parser` for the text-source seam). A task family with a genuinely different output structure, such as seq2seq, still needs its own `Data`-layer work (tracked in GH issue #285); this section only covers what today's loader already handles.

## Worked example: detection

`DetectionFamily` (`src/raitap/task_families/detection.py`) is the complete reference for a non-classification family: `fixed_output_space = DETECTION_BOXES`, `adapt_loaded_inputs` unbinds to a ragged list, `load_labels` parses a JSON records file aligned by `sample_id`, `explain` drives a per-box K-loop via `explain_detection`, `supports_robustness` returns `False` (Phase 4 follow-up), and `matches_model` recognises torchvision detectors. Its runnable end-to-end config lives at `contributor-configs/fasterrcnn-udacity/`.
