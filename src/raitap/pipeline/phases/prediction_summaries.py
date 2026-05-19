"""Prediction post-processing — per-sample ``PredictionSummary`` rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raitap.pipeline.outputs import ForwardOutput, PredictionSummary
from raitap.types import TaskKind
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
    forward_output: ForwardOutput,
    sample_ids: list[str] | None,
    targets: torch.Tensor | list[dict[str, torch.Tensor]] | None,
) -> tuple[PredictionSummary, ...]:
    """Reduce classification logits to per-sample ``PredictionSummary`` rows."""
    if forward_output.task_kind is not TaskKind.classification:
        # Detection / regression / etc. don't have a "predicted class +
        # confidence" concept per sample. Reporting handles empty.
        return ()
    predictions_tensor = forward_output.predictions_tensor
    assert predictions_tensor is not None  # invariant from ForwardOutput.__post_init__
    if predictions_tensor.ndim != 2 or predictions_tensor.shape[1] < 2:
        return ()

    classification_targets = targets if not isinstance(targets, list) else None
    probabilities = torch.softmax(predictions_tensor.detach().cpu(), dim=1)
    confidences, predictions = probabilities.max(dim=1)
    resolved_targets = valid_targets_for_reporting(
        targets=classification_targets,
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
