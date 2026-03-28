"""Assessment pipeline (model forward, metrics, explainers)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from raitap.configs import resolve_run_dir
from raitap.data import Data
from raitap.metrics import Metrics, MetricsEvaluation, metrics_run_enabled
from raitap.models import Model
from raitap.run.forward_output import extract_primary_tensor
from raitap.run.metrics_placeholder import metrics_prediction_pair
from raitap.run.outputs import RunOutputs
from raitap.tracking import BaseTracker
from raitap.transparency.factory import Explanation

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig
    from raitap.transparency.results import ExplanationResult, VisualisationResult


def run(config: AppConfig) -> RunOutputs:
    model = Model(config)
    data = Data(config)

    outputs = _run_without_tracking(config, model, data)

    has_tracker = bool(str(config.tracking._target_).strip())
    if not has_tracker:
        return outputs

    use_subdirs = len(outputs.explanations) > 1
    with BaseTracker.create_tracker(config) as tracker:
        tracker.log_config()
        if config.tracking.log_model:
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
        forward_output = extract_primary_tensor(raw_output)

    metrics_eval: MetricsEvaluation | None = None
    if metrics_run_enabled(config):
        if (
            config.metrics.num_classes is None
            and forward_output.ndim == 2
            and forward_output.shape[1] >= 2
        ):
            config.metrics.num_classes = int(forward_output.shape[1])
        preds, targs = metrics_prediction_pair(forward_output)
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
