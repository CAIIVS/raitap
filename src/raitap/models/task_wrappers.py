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
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must lie in [0, 1]; got {iou_threshold!r}.")
        if mode == "reference_match":
            if reference_xyxy is None:
                raise ValueError("DetectionTarget(mode='reference_match') requires reference_xyxy.")
            if reference_label is None:
                raise ValueError(
                    "DetectionTarget(mode='reference_match') requires reference_label."
                )
        self.mode = mode
        self.box_idx = int(box_idx)
        self.reference_xyxy = (
            tuple(float(x) for x in reference_xyxy) if reference_xyxy is not None else None
        )
        self.reference_label = int(reference_label) if reference_label is not None else None
        self.iou_threshold = float(iou_threshold)

    def __call__(self, model_out: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        if not isinstance(model_out, list) or (model_out and not isinstance(model_out[0], dict)):
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
                # Keep the autograd graph alive when at least one tensor is
                # present so gradient explainers (Captum / SHAP gradient /
                # Grad-CAM) get a non-leaf zero. Pure-None case falls back
                # to a detached leaf (unreachable in practice — a detector
                # that returns an empty dict shouldn't be wrapped).
                live_tensor = next((t for t in (scores, boxes, labels) if t is not None), None)
                if live_tensor is None:
                    per_sample_scores.append(torch.tensor(0.0))
                else:
                    per_sample_scores.append(live_tensor.sum() * 0.0)
                continue
            reference = torch.tensor(self.reference_xyxy, device=boxes.device, dtype=boxes.dtype)
            ious = _box_iou(reference, boxes)
            label_mask = labels == self.reference_label
            iou_mask = ious >= self.iou_threshold
            combined = label_mask & iou_mask
            if not combined.any():
                # Differentiable zero tied to scores so the autograd graph
                # survives a no-match step (perturbed input might drift the
                # detection away from the reference box for one step but
                # back in a later step — gradient explainers must still
                # propagate through scores).
                per_sample_scores.append(scores.sum() * 0.0)
                continue
            masked_ious = torch.where(combined, ious, torch.full_like(ious, -1.0))
            best_idx = int(torch.argmax(masked_ious).item())
            per_sample_scores.append(scores[best_idx])
        return torch.stack(per_sample_scores).sum()


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
