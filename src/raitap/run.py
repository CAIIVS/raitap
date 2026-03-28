from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import hydra
import torch

from raitap.configs import register_configs, resolve_run_dir
from raitap.data import Data
from raitap.metrics import Metrics, MetricsEvaluation, metrics_run_enabled
from raitap.models import Model
from raitap.tracking import BaseTracker
from raitap.transparency.factory import Explanation

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig
    from raitap.transparency.results import ExplanationResult, VisualisationResult

register_configs()
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: AppConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print_summary(config)
    run(config)

    logger.info("\n%s", "=" * 60)
    logger.info("Assessment complete!")
    logger.info("%s", "=" * 60)


@dataclass(frozen=True)
class RunOutputs:
    """Assessment outputs; ``forward_output`` is the primary tensor from ``model(data)``."""

    explanations: list[ExplanationResult]
    visualisations: list[VisualisationResult]
    metrics: MetricsEvaluation | None
    forward_output: torch.Tensor


def run(config: AppConfig) -> RunOutputs:
    model = Model(config)
    data = Data(config)

    outputs = _run_without_tracking(config, model, data)

    tracking_config = getattr(config, "tracking", None)
    has_tracker = bool(tracking_config and getattr(tracking_config, "_target_", None))
    if not has_tracker:
        return outputs

    use_subdirs = len(outputs.explanations) > 1
    with BaseTracker.create_tracker(config) as tracker:
        tracker.log_config()
        if getattr(config.tracking, "log_model", False):
            model.log(tracker)
        data.log(tracker)
        if outputs.metrics is not None:
            outputs.metrics.log(tracker)
        for explanation in outputs.explanations:
            explanation.log(tracker, use_subdirectory=use_subdirs)
        for visualisation in outputs.visualisations:
            visualisation.log(tracker, use_subdirectory=use_subdirs)

    return outputs


def _run_without_tracking(config: AppConfig, model: Model, data: Data) -> RunOutputs:
    data_tensor = data.tensor

    with torch.no_grad():
        raw_output: Any = model.network(data_tensor)
        forward_output = _extract_primary_tensor(raw_output)

    metrics_eval: MetricsEvaluation | None = None
    if metrics_run_enabled(config):
        if (
            getattr(config.metrics, "num_classes", None) is None
            and forward_output.ndim == 2
            and forward_output.shape[1] >= 2
        ):
            config.metrics.num_classes = int(forward_output.shape[1])
        preds, targs = _metrics_prediction_pair(forward_output)
        # No labels in DataConfig yet: placeholder pairing; configure metrics to match your task.
        metrics_eval = Metrics(config, preds, targs)

    explanations: list[ExplanationResult] = []
    visualisations: list[VisualisationResult] = []

    explainers = config.transparency.items()
    if not explainers:
        raise ValueError("No explainers configured")

    for name, _explainer_cfg in explainers:
        # Explainer ``call:`` (and optional run kwargs) supply target, baselines, etc.
        explanation = Explanation(config, name, model, data_tensor)
        explanations.append(explanation)
        visualisations.extend(explanation.visualise())

    return RunOutputs(
        explanations=explanations,
        visualisations=visualisations,
        metrics=metrics_eval,
        forward_output=forward_output,
    )


def _extract_primary_tensor(model_output: object) -> torch.Tensor:
    """
    Unwrap a single tensor from common forward return shapes.

    Supports ``Tensor``, the first ``Tensor`` in a ``tuple`` / ``list``, or a ``dict``
    (keys tried: ``logits``, ``pred``, ``prediction``, ``output``, ``scores``, then any value).
    """
    if isinstance(model_output, torch.Tensor):
        return model_output

    if isinstance(model_output, (tuple, list)):
        for item in model_output:
            if isinstance(item, torch.Tensor):
                return item
        raise TypeError("Model forward returned a sequence with no torch.Tensor elements.")

    if isinstance(model_output, dict):
        for key in ("logits", "pred", "prediction", "output", "scores"):
            value = model_output.get(key)
            if isinstance(value, torch.Tensor):
                return value
        for value in model_output.values():
            if isinstance(value, torch.Tensor):
                return value
        raise TypeError("Model forward returned a dict with no torch.Tensor values.")

    raise TypeError(
        f"Unsupported model output type {type(model_output).__name__!r}; "
        "expected Tensor, sequence of Tensors, or dict containing a Tensor."
    )


def _metrics_prediction_pair(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def print_summary(config: AppConfig) -> None:
    logger.info("%s", "=" * 60)
    logger.info("RAITAP Transparency Assessment")
    logger.info("%s", "=" * 60)
    logger.info("\nExperiment: %s", config.experiment_name)
    logger.info("Model: %s", config.model.source)
    logger.info("Dataset: %s", config.data.name)
    logger.info("Explainers: %s", list(config.transparency.keys()))
    logger.info("Metrics: %s", "on" if metrics_run_enabled(config) else "off")
    logger.info("Output: %s\n", resolve_run_dir(config))


if __name__ == "__main__":
    main()
