"""Task-generic helpers that map structured model outputs to a scalar.

Detection models (e.g. ``fasterrcnn_resnet50_fpn_v2``) return
``list[dict[str, Tensor]]``. The existing transparency explainers and
robustness adversarial attacks assume a scalar-per-sample output. This
module bridges that gap with two pieces:

* :class:`DetectionTarget` — reduces a detection model's output to a
  single ``torch.Tensor`` scalar via one of three modes
  (``class_score`` / ``objectness`` / ``bbox_l2``).
* :class:`ScalarDetectionWrapper` — added in a follow-up task; wraps a
  detection model so existing scalar-output adapters can consume it.
"""

from __future__ import annotations

from typing import Literal

import torch

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

        # mode == "bbox_l2"
        boxes = model_out[0].get("boxes")
        if boxes is None or boxes.numel() == 0 or boxes.shape[0] <= self.box_idx:
            return torch.tensor(0.0)
        return (boxes[self.box_idx] ** 2).sum()
