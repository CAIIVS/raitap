"""Robustness phase — instantiates assessors + resolves attack targets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raitap import raitap_log
from raitap.metrics import metrics_prediction_pair
from raitap.robustness.factory import RobustnessAssessment

if TYPE_CHECKING:
    import torch

    from raitap.configs.schema import AppConfig
    from raitap.data import Data
    from raitap.models import Model
    from raitap.robustness.results import RobustnessResult, RobustnessVisualisationResult
    from raitap.transparency.contracts import InputSpec


def resolve_robustness_targets(
    *,
    labels: torch.Tensor | None,
    forward_output: torch.Tensor,
) -> torch.Tensor | None:
    """Return per-sample reference labels for robustness assessors.

    Mirrors the metrics fallback (see :func:`resolve_metric_targets`): when the
    data pipeline supplies ground-truth labels we use them; otherwise we fall
    back to ``argmax(model(clean))`` so an untargeted attack still has a
    well-defined reference (the attack tries to push the model away from its
    current decision). Returns ``None`` only when neither labels nor a usable
    classification head are available, in which case the assessor will raise
    :class:`MissingTargetsError`.
    """
    if labels is not None:
        return labels
    if forward_output.ndim != 2 or forward_output.shape[1] < 2:
        return None
    predictions, _ = metrics_prediction_pair(forward_output)

    from raitap.utils.diagnostics import Module

    raitap_log.warn(
        "No ground-truth labels provided; using model predictions as the "
        "reference for untargeted attacks.",
        module=Module.robustness,
    )
    return predictions.detach().cpu()


def assess_robustness(
    config: AppConfig,
    model: Model,
    data: Data,
    forward_output: torch.Tensor,
    *,
    labels: torch.Tensor | None,
    input_metadata: InputSpec | None,
) -> tuple[list[RobustnessResult], list[RobustnessVisualisationResult]]:
    """Run every assessor declared under ``config.robustness``.

    Returns ``(results, visualisations)``. Each result's ``visualise()`` output
    is flattened into the visualisations list.
    """
    assessors = getattr(config, "robustness", None) or {}
    if not assessors:
        return [], []

    suffix = "s" if len(assessors) > 1 else ""
    raitap_log.info("Performing robustness assessment%s (%d)...", suffix, len(assessors))

    targets = resolve_robustness_targets(labels=labels, forward_output=forward_output)
    results: list[RobustnessResult] = []
    visualisations: list[RobustnessVisualisationResult] = []
    for name in assessors:
        result = RobustnessAssessment(
            config,
            name,
            model,
            data.tensor,
            targets,
            input_metadata=input_metadata,
            sample_ids=data.sample_ids,
            sample_names=data.sample_ids,
        )
        results.append(result)
        visualisations.extend(result.visualise())
    return results, visualisations
