"""Prepare metric predictions/targets from model outputs and optional labels."""

from __future__ import annotations

import warnings
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


def resolve_metric_targets(
    predictions: torch.Tensor,
    labels: torch.Tensor | None,
) -> torch.Tensor:
    """Use ground truth labels when available, else warn and fall back to predictions."""
    if labels is None:
        warnings.warn(
            "No ground-truth labels provided; falling back to predictions as metric targets.",
            stacklevel=2,
        )
        return predictions

    if labels.shape[0] != predictions.shape[0]:
        warnings.warn(
            "Ground-truth labels do not match prediction count; "
            "falling back to predictions as metric targets.",
            stacklevel=2,
        )
        return predictions

    return labels