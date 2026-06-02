"""Pure helpers for detection box labelling (issue #233).

No I/O, no model, no GPU — keeps id->name resolution and box->GT matching
unit-testable on synthetic tensors. ``resolve_category_names`` fixes source
precedence (explicit config > backend weights.meta > None) in one place.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from raitap.transparency.contracts import DetectionBox


def resolve_category_names(
    explicit: Sequence[str] | None,
    backend_categories: Sequence[str] | None,
) -> list[str] | None:
    """Resolve the id->name table. Precedence: explicit config > backend > None."""
    if explicit is not None:
        return list(explicit)
    if backend_categories is not None:
        return list(backend_categories)
    return None


def label_name_for(index: int, names: Sequence[str] | None) -> str | None:
    """Bounds-safe id->name lookup; ``None`` when no map or index out of range."""
    if names is None:
        return None
    if 0 <= index < len(names):
        return names[index]
    return None


def enrich_detection_box(
    box: DetectionBox,
    *,
    category_names: Sequence[str] | None = None,
    ground_truth_for_sample: dict[str, torch.Tensor] | None = None,
    iou_threshold: float = 0.5,
) -> DetectionBox:
    """Return *box* with display metadata resolved (predicted name + GT match).

    Pure: no model, no I/O. ``explain_detection`` builds the raw box (geometry +
    ``label_index``); the transparency caller calls this before rendering.

    - ``label_name`` is resolved from ``category_names`` (``None`` -> numeric id
      fallback downstream).
    - When ``ground_truth_for_sample`` is given (the sample's GT dict with ``boxes``/
      ``labels``), the box is matched to GT by class-agnostic IoU and
      ``ground_truth_evaluated`` is set ``True`` regardless of whether a match was found.
      A match fills ``true_label_index`` / ``true_label_name`` / ``true_match_iou``;
      no match leaves them ``None`` (a false positive). ``ground_truth_for_sample=None``
      means GT was not configured and leaves ``ground_truth_evaluated`` ``False``.
    """
    true_label_index: int | None = None
    true_label_name: str | None = None
    true_match_iou: float | None = None
    if ground_truth_for_sample is not None:
        match = match_box_to_ground_truth(
            box.xyxy,
            ground_truth_for_sample["boxes"],
            ground_truth_for_sample["labels"],
            iou_threshold,
        )
        if match is not None:
            true_label_index, true_match_iou = match
            true_label_name = label_name_for(true_label_index, category_names)
    return dataclasses.replace(
        box,
        label_name=label_name_for(box.label_index, category_names),
        ground_truth_evaluated=ground_truth_for_sample is not None,
        true_label_index=true_label_index,
        true_label_name=true_label_name,
        true_match_iou=true_match_iou,
    )


def match_box_to_ground_truth(
    pred_xyxy: tuple[float, float, float, float],
    ground_truth_boxes: torch.Tensor,
    ground_truth_labels: torch.Tensor,
    iou_threshold: float,
) -> tuple[int, float] | None:
    """Class-agnostic best-IoU match of a predicted box to ground truth.

    Returns ``(ground_truth_label_index, iou)`` for the highest-IoU GT box at or above
    ``iou_threshold``, else ``None`` (no match -> the prediction is a false
    positive). Matching ignores class so "predicted X, truth Y" disagreements
    surface; the GT class is reported, not required to equal the prediction.

    ``ground_truth_boxes`` is ``(M, 4)`` xyxy and ``ground_truth_labels`` is ``(M,)`` int, both in
    the same pixel space as ``pred_xyxy`` (the detection forward-pass space).

    ``torch`` / ``torchvision`` are imported lazily here so the module's pure
    name-resolution helpers stay importable without the optional torch extra.
    """
    import torch
    from torchvision.ops import box_iou

    if ground_truth_boxes.numel() == 0:
        return None
    pred = torch.tensor(
        [pred_xyxy], dtype=ground_truth_boxes.dtype, device=ground_truth_boxes.device
    )
    ious = box_iou(pred, ground_truth_boxes)[0]  # (M,)
    best = int(ious.argmax().item())
    best_iou = float(ious[best].item())
    if best_iou < iou_threshold:
        return None
    return int(ground_truth_labels[best].item()), best_iou
