"""Assessment pipeline (model forward, metrics, explainers)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from raitap.configs import cfg_to_dict, resolve_run_dir
from raitap.data import Data
from raitap.metrics import (
    Metrics,
    MetricsEvaluation,
    metrics_prediction_pair,
    metrics_run_enabled,
    resolve_metric_targets,
)
from raitap.models import Model
from raitap.run.forward_output import extract_primary_tensor
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
    print_summary(config, model)

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
    data_tensor = model.backend._prepare_inputs(data.tensor)

    with torch.no_grad():
        raw_output: Any = model.backend(data_tensor)
        forward_output = extract_primary_tensor(raw_output)

    metrics_eval: MetricsEvaluation | None = None
    if metrics_run_enabled(config):
        if (
            getattr(config.metrics, "num_classes", None) is None
            and forward_output.ndim == 2
            and forward_output.shape[1] >= 2
        ):
            config.metrics.num_classes = int(forward_output.shape[1])
        preds, _ = metrics_prediction_pair(forward_output)
        targs = resolve_metric_targets(preds, getattr(data, "labels", None))
        metrics_eval = Metrics(config, preds, targs)

    explanations: list[ExplanationResult] = []
    visualisations: list[VisualisationResult] = []

    explainers = config.transparency.items()
    if not explainers:
        raise ValueError("No explainers configured")

    for name, _explainer_cfg in explainers:
        runtime_kwargs = _resolve_explainer_runtime_kwargs(
            config.transparency[name],
            forward_output=forward_output,
        )
        # Explainer ``call:`` (and optional run kwargs) supply target, baselines, etc.
        explanation = Explanation(
            config,
            name,
            model,
            data_tensor,
            sample_names=getattr(data, "sample_ids", None),
            **runtime_kwargs,
        )
        explanations.append(explanation)
        visualisations.extend(explanation.visualise())

    return RunOutputs(
        explanations=explanations,
        visualisations=visualisations,
        metrics=metrics_eval,
        forward_output=forward_output,
    )


def print_summary(config: AppConfig, model: Model) -> None:
    logger.info("%s", "=" * 60)
    logger.info("RAITAP Assessment")
    logger.info("%s", "=" * 60)
    logger.info("\nExperiment: %s", config.experiment_name)
    logger.info("Model: %s", config.model.source)
    logger.info("Dataset: %s", config.data.name)
    logger.info("Hardware: %s", model.backend.hardware_label)
    logger.info("Explainers: %s", list(config.transparency.keys()))
    logger.info("Metrics: %s", "on" if metrics_run_enabled(config) else "off")
    logger.info("Output: %s\n", resolve_run_dir(config))


def _resolve_explainer_runtime_kwargs(
    explainer_config: Any,
    *,
    forward_output: torch.Tensor,
) -> dict[str, Any]:
    raw_config = cfg_to_dict(explainer_config)
    call_config = raw_config.get("call")
    if not isinstance(call_config, dict):
        return {}

    if call_config.get("target") != "auto_pred":
        return {}

    predictions, _ = metrics_prediction_pair(forward_output)
    return {"target": predictions}
