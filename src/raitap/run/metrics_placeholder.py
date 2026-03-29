"""Placeholder (pred, target) tensors when the pipeline has no ground-truth labels."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def metrics_prediction_pair(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a placeholder (predictions, targets) pair for metrics when no labels exist.

    For multiclass logits ``(N, C)`` with ``C > 1``, uses ``argmax`` for both (trivial
    self-consistency). For other shapes, passes ``output`` through unchanged so users
    can pair metrics configs with regression / detection / etc.
    """
    if output.ndim == 2 and output.shape[1] > 1:
        labels = output.argmax(dim=1)
        return labels, labels
    if output.ndim == 2 and output.shape[1] == 1:
        squeezed = output.squeeze(1)
        return squeezed, squeezed
    return output, output
