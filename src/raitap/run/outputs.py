"""Structured return value from :func:`~raitap.run.run`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from raitap.metrics import MetricsEvaluation
    from raitap.transparency.results import ExplanationResult, VisualisationResult


@dataclass(frozen=True)
class RunOutputs:
    """Assessment outputs; ``forward_output`` is the primary tensor from ``model(data)``."""

    explanations: list[ExplanationResult]
    visualisations: list[VisualisationResult]
    metrics: MetricsEvaluation | None
    forward_output: torch.Tensor
