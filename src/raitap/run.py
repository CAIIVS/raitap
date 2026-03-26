from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    explanations: list[ExplanationResult]
    visualisations: list[VisualisationResult]
    metrics: MetricsEvaluation | None
    predicted_classes: torch.Tensor


def _run_without_tracking(config: AppConfig, model: Model, data: Data) -> RunOutputs:
    data_tensor = data.tensor

    with torch.no_grad():
        logits = model.network(data_tensor)
        predicted_classes = logits.argmax(dim=1)
        target = predicted_classes.tolist()

    metrics_eval: MetricsEvaluation | None = None
    if metrics_run_enabled(config):
        if getattr(config.metrics, "num_classes", None) is None and logits.ndim >= 2:
            config.metrics.num_classes = int(logits.shape[1])
        # No labels in DataConfig yet: predictions vs themselves yields a trivial self-consistency
        # check only; replace with real targets when the pipeline exposes ground truth.
        metrics_eval = Metrics(config, predicted_classes, predicted_classes)

    explanations: list[ExplanationResult] = []
    visualisations: list[VisualisationResult] = []

    explainers = config.transparency.items()
    if not explainers:
        raise ValueError("No explainers configured")

    for name, _explainer_cfg in explainers:
        explanation = Explanation(config, name, model, data_tensor, target=target)
        explanations.append(explanation)
        visualisations.extend(explanation.visualise())

    return RunOutputs(
        explanations=explanations,
        visualisations=visualisations,
        metrics=metrics_eval,
        predicted_classes=predicted_classes,
    )


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
