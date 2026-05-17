"""Prediction post-processing — per-sample ``PredictionSummary`` rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raitap.pipeline.outputs import PredictionSummary
from raitap.utils.lazy import lazy_import

if TYPE_CHECKING:
    import torch
else:
    torch = lazy_import("torch")


def valid_targets_for_reporting(
    *,
    targets: torch.Tensor | None,
    expected: int,
) -> torch.Tensor | None:
    """Return a CPU 1-D target tensor matching the expected length, else ``None``."""
    if targets is None:
        return None
    if targets.ndim != 1 or int(targets.shape[0]) != expected:
        return None
    return targets.detach().cpu()


def prediction_summaries(
    *,
    forward_output: torch.Tensor,
    sample_ids: list[str] | None,
    targets: torch.Tensor | None,
) -> tuple[PredictionSummary, ...]:
    """Reduce classification logits to per-sample ``PredictionSummary`` rows."""
    if forward_output.ndim != 2 or forward_output.shape[1] < 2:
        return ()

    probabilities = torch.softmax(forward_output.detach().cpu(), dim=1)
    confidences, predictions = probabilities.max(dim=1)
    resolved_targets = valid_targets_for_reporting(
        targets=targets,
        expected=int(predictions.shape[0]),
    )

    summaries: list[PredictionSummary] = []
    names = [] if sample_ids is None else [str(item) for item in sample_ids]
    pairs = zip(predictions, confidences, strict=False)
    for index, (predicted_class, confidence) in enumerate(pairs):
        target_class: int | None = None
        correct: bool | None = None
        if resolved_targets is not None:
            target_class = int(resolved_targets[index].item())
            correct = int(predicted_class.item()) == target_class
        summaries.append(
            PredictionSummary(
                sample_index=index,
                sample_id=names[index] if index < len(names) else None,
                predicted_class=int(predicted_class.item()),
                target_class=target_class,
                confidence=float(confidence.item()),
                correct=correct,
            )
        )
    return tuple(summaries)
