"""Robustness assessment phase — instantiates assessors + resolves attack targets.

Co-located with the module it drives (issue #243 follow-up): the phase class,
its work function, and target resolution live here; the result type +
report rendering live in :mod:`raitap.robustness.report`. The pipeline only
assembles ``RobustnessPhase`` into the registry — it owns none of this logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from raitap import raitap_log
from raitap.metrics import metrics_prediction_pair
from raitap.pipeline.phases.base import AssessmentPhase, run_adapters
from raitap.robustness.factory import RobustnessAssessment
from raitap.robustness.report import RobustnessPhaseResult
from raitap.types import TaskKind

if TYPE_CHECKING:
    import torch

    from raitap.configs.schema import AppConfig
    from raitap.data import Data
    from raitap.data.preprocessing import ResolvedPreprocessing
    from raitap.models import Model
    from raitap.pipeline.outputs import ForwardOutput, PhaseResult
    from raitap.pipeline.phases.base import PhaseContext
    from raitap.robustness.results import RobustnessResult
    from raitap.transparency.contracts import InputSpec


class RobustnessPhase(AssessmentPhase):
    name = "robustness"

    def is_configured(self, config: AppConfig) -> bool:
        return bool(getattr(config, "robustness", None))

    def run(self, ctx: PhaseContext) -> PhaseResult | None:
        results = assess_robustness(
            ctx.config,
            ctx.model,
            ctx.data,
            ctx.forward_output,
            labels=ctx.data.labels,
            input_metadata=ctx.input_metadata,
            resolved_preprocessing=ctx.resolved_preprocessing,
        )
        return RobustnessPhaseResult(results=results)


def resolve_robustness_targets(
    *,
    labels: torch.Tensor | None,
    forward_output: ForwardOutput,
) -> torch.Tensor | None:
    """Return per-sample reference labels for robustness assessors.

    Mirrors the metrics fallback (see :func:`resolve_metric_targets`): when the
    data pipeline supplies ground-truth labels we use them; otherwise we fall
    back to ``argmax(model(clean))`` so an untargeted attack still has a
    well-defined reference (the attack tries to push the model away from its
    current decision). Detection backends don't expose a single per-sample
    class — we always fall back to ``None`` for them; the assessor will raise
    :class:`MissingTargetsError` if it needs labels.
    """
    if labels is not None:
        return labels
    if forward_output.task_kind is not TaskKind.classification:
        return None
    predictions_tensor = forward_output.predictions_tensor
    assert predictions_tensor is not None
    if predictions_tensor.ndim != 2 or predictions_tensor.shape[1] < 2:
        return None
    predictions, _ = metrics_prediction_pair(predictions_tensor)

    # Module chip is inferred from the call site (this file lives under
    # ``robustness/``), so no explicit ``module=`` is needed.
    raitap_log.warn(
        "No ground-truth labels provided; using model predictions as the "
        "reference for untargeted attacks.",
    )
    return predictions.detach().cpu()


def assess_robustness(
    config: AppConfig,
    model: Model,
    data: Data,
    forward_output: ForwardOutput,
    *,
    labels: torch.Tensor | list[dict[str, torch.Tensor]] | None,
    input_metadata: InputSpec | None,
    resolved_preprocessing: ResolvedPreprocessing | None = None,
) -> list[RobustnessResult]:
    """Run every assessor declared under ``config.robustness``.

    Returns the assessor results. Each result owns its report visualisations
    (``RobustnessResult.visualisations``), populated via the shared
    :func:`~raitap.pipeline.phases.base.run_adapters` loop.
    """
    if forward_output.task_kind is not TaskKind.classification:
        # Robustness against detection models is a Phase 4 deliverable
        # (DetectionAdversarialLoss). Empirical / formal robustness in this
        # phase supports classification only.
        return []

    assessors = getattr(config, "robustness", None) or {}
    classification_labels = labels if not isinstance(labels, list) else None
    targets = resolve_robustness_targets(
        labels=classification_labels, forward_output=forward_output
    )
    # Non-classification short-circuited above, so this is the dense NCHW
    # classification path; narrow the ``Data.tensor`` union.
    tensor = cast("torch.Tensor", data.tensor)
    return run_adapters(
        assessors,
        log_label="robustness",
        build_one=lambda name: RobustnessAssessment(
            config,
            name,
            model,
            tensor,
            targets,
            input_metadata=input_metadata,
            sample_ids=data.sample_ids,
            sample_names=data.sample_ids,
            resolved_preprocessing=resolved_preprocessing,
        ),
    )
