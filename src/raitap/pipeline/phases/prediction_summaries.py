"""Prediction post-processing — per-sample ``PredictionSummary`` rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raitap.utils.lazy import lazy_import

if TYPE_CHECKING:
    import torch

    from raitap.pipeline.outputs import ForwardOutput, PredictionSummary
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
    """Delegate per-sample ``PredictionSummary`` rows to the task family."""
    from raitap.task_families import resolve_task_family

    family = resolve_task_family(forward_output.task_kind)
    rows = family.prediction_summaries(
        forward_output.payload,
        sample_ids=sample_ids,
        targets=targets,
        output_kind=forward_output.output_kind,
    )
    return tuple(rows) if rows is not None else ()
