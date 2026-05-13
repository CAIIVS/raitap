"""Assessment pipeline (model forward, metrics, explainers)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import torch

from raitap import raitap_log
from raitap.configs import cfg_to_dict
from raitap.data import Data, infer_data_input_metadata
from raitap.metrics import (
    Metrics,
    MetricsEvaluation,
    metrics_prediction_pair,
    metrics_run_enabled,
    resolve_metric_targets,
)
from raitap.models import Model
from raitap.reporting import (
    build_report,
    create_report,
    reporting_enabled,
)
from raitap.reporting.sample_selection import resolve_report_sample_selection
from raitap.robustness.factory import RobustnessAssessment
from raitap.run.forward_output import extract_primary_tensor
from raitap.run.outputs import PredictionSummary, RunOutputs
from raitap.tracking import BaseTracker
from raitap.transparency.contracts import InputSpec
from raitap.transparency.factory import Explanation

# Conservative default for prediction/metrics forwards. Transparency methods have their own
# per-explainer ``transparency.*.raitap.batch_size`` controls.
_DEFAULT_FORWARD_BATCH_SIZE = 32


if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig
    from raitap.robustness.results import RobustnessResult, RobustnessVisualisationResult
    from raitap.transparency.results import ExplanationResult, VisualisationResult


def _log_phase_start(phase: str, n: int) -> None:
    suffix = "s" if n > 1 else ""
    raitap_log.info("Performing %s%s (%d)...", phase, suffix, n)


def run(config: AppConfig) -> RunOutputs:
    # Capture warnings emitted during model + data construction so the
    # summary panel renders first; otherwise the rich handler interleaves
    # them above the banner and makes the run header look fragmented.
    with warnings.catch_warnings(record=True) as deferred:
        warnings.simplefilter("always")
        model = Model(config)
        data = Data(config)
    _validate_report_sample_selection(config, data)
    print_summary(config, model)
    for entry in deferred:
        warnings.warn_explicit(
            entry.message,
            entry.category,
            entry.filename,
            entry.lineno,
            source=entry.source,
        )

    outputs = _run_without_tracking(config, model, data)

    # Generate report if configured
    report_generation = None
    if reporting_enabled(config):
        raitap_log.info("Generating report...")
        report = build_report(config, outputs)
        report_generation = create_report(config=config, report=report)

    tracking_config = getattr(config, "tracking", None)
    has_tracker = bool(tracking_config and getattr(tracking_config, "_target_", None))
    if not has_tracker:
        return outputs

    use_subdirs = len(outputs.explanations) > 1
    use_robustness_subdirs = len(outputs.robustness_results) > 1
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
        for robustness_result in outputs.robustness_results:
            robustness_result.log(tracker, use_subdirectory=use_robustness_subdirs)
        for robustness_visualisation in outputs.robustness_visualisations:
            robustness_visualisation.log(tracker, use_subdirectory=use_robustness_subdirs)
        # Log report to tracker
        reporting_cfg = getattr(config, "reporting", None)
        if report_generation is not None and reporting_cfg is not None:
            report_generation.log(tracker)

    return outputs


def _validate_report_sample_selection(config: AppConfig, data: Data) -> None:
    if not reporting_enabled(config):
        return
    reporting_cfg = config.reporting
    selection = None if reporting_cfg is None else getattr(reporting_cfg, "sample_selection", None)
    resolve_report_sample_selection(
        selection,
        sample_ids=data.sample_ids,
        batch_size=int(data.tensor.shape[0]) if data.tensor.ndim > 0 else 0,
    )


def _run_without_tracking(config: AppConfig, model: Model, data: Data) -> RunOutputs:
    backend = model.backend
    data_tensor = data.tensor
    sample_ids = data.sample_ids
    labels = data.labels

    raitap_log.info("Running model forward pass...")
    with torch.no_grad():
        forward_output = _forward_primary_tensor(config, backend, data_tensor)

    metrics_eval: MetricsEvaluation | None = None
    if metrics_run_enabled(config):
        raitap_log.info("Computing metrics...")
        if (
            getattr(config.metrics, "num_classes", None) is None
            and forward_output.ndim == 2
            and forward_output.shape[1] >= 2
        ):
            config.metrics.num_classes = int(forward_output.shape[1])
        preds, _ = metrics_prediction_pair(forward_output)
        targs = resolve_metric_targets(preds, labels)
        metrics_eval = Metrics(config, preds, targs)

    explanations: list[ExplanationResult] = []
    visualisations: list[VisualisationResult] = []

    explainers = list((getattr(config, "transparency", None) or {}).items())
    robustness_assessors = getattr(config, "robustness", None) or {}

    if not explainers and not robustness_assessors:
        raise ValueError("No explainers or robustness assessors configured")

    if explainers:
        _log_phase_start("transparency assessment", len(explainers))
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
            input_metadata=_input_metadata_for_data(config, data),
            sample_ids=sample_ids,
            sample_names=sample_ids,
            **runtime_kwargs,
        )
        explanations.append(explanation)
        visualisations.extend(explanation.visualise())

    robustness_results: list[RobustnessResult] = []
    robustness_visualisations: list[RobustnessVisualisationResult] = []
    if robustness_assessors:
        _log_phase_start("robustness assessment", len(robustness_assessors))
    robustness_targets = _robustness_targets(labels=labels, forward_output=forward_output)
    for name in robustness_assessors:
        result = RobustnessAssessment(
            config,
            name,
            model,
            data_tensor,
            robustness_targets,
            input_metadata=_input_metadata_for_data(config, data),
            sample_ids=sample_ids,
            sample_names=sample_ids,
        )
        robustness_results.append(result)
        robustness_visualisations.extend(result.visualise())

    return RunOutputs(
        explanations=explanations,
        visualisations=visualisations,
        metrics=metrics_eval,
        forward_output=forward_output.detach().cpu(),
        sample_ids=sample_ids,
        targets=labels,
        prediction_summaries=_prediction_summaries(
            forward_output=forward_output,
            sample_ids=sample_ids,
            targets=labels,
        ),
        robustness_results=robustness_results,
        robustness_visualisations=robustness_visualisations,
    )


def _forward_primary_tensor(config: AppConfig, backend: Any, inputs: torch.Tensor) -> torch.Tensor:
    batch_size = _resolve_forward_batch_size(config)
    total_batch = int(inputs.shape[0])
    if total_batch <= batch_size:
        prepared_inputs = backend._prepare_inputs(inputs)
        raw_output: Any = backend(prepared_inputs)
        forward_output = extract_primary_tensor(raw_output).detach().cpu()
        del prepared_inputs, raw_output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return forward_output

    chunks: list[torch.Tensor] = []
    for start in range(0, total_batch, batch_size):
        end = min(start + batch_size, total_batch)
        prepared_inputs = backend._prepare_inputs(inputs[start:end])
        raw_output = backend(prepared_inputs)
        chunks.append(extract_primary_tensor(raw_output).detach().cpu())
        del prepared_inputs, raw_output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def _resolve_forward_batch_size(config: AppConfig) -> int:
    """Resolve prediction/metrics forward batch size from config, falling back to 32."""
    configured = getattr(getattr(config, "run", None), "forward_batch_size", None)
    if configured is None:
        configured = getattr(getattr(config, "data", None), "forward_batch_size", None)
    if configured is None:
        return _DEFAULT_FORWARD_BATCH_SIZE
    if not isinstance(configured, int):
        raise TypeError(f"forward_batch_size must be an int, got {type(configured).__name__}.")
    if configured <= 0:
        raise ValueError(f"forward_batch_size must be > 0, got {configured}.")
    return configured


def print_summary(config: AppConfig, model: Model) -> None:
    from raitap.utils.console import print_summary_panel

    print_summary_panel(config, model)


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
    return {"target": predictions.detach()}


def _robustness_targets(
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

    from raitap.utils.diagnostics import Subsystem

    raitap_log.warn(
        "No ground-truth labels provided; using model predictions as the "
        "reference for untargeted attacks.",
        subsystem=Subsystem.robustness,
    )
    return predictions.detach().cpu()


def _input_metadata_for_data(config: AppConfig, data: Data) -> InputSpec | None:
    """Return runtime input metadata derived from the data object, or ``None``
    if neither ``kind`` nor ``layout`` can be determined (in which case any
    ``transparency.<explainer>.raitap.input_metadata`` from the explainer
    config will be used unchanged)."""
    explicit = getattr(data, "input_metadata", None)
    if isinstance(explicit, InputSpec):
        return explicit
    config_explicit = getattr(getattr(config, "data", None), "input_metadata", None)
    if isinstance(config_explicit, InputSpec):
        return config_explicit
    metadata = infer_data_input_metadata(config, data)
    if metadata.kind is None and metadata.layout is None:
        # Don't override yaml-provided ``raitap.input_metadata`` with an empty
        # spec — let the explainer-level config drive output-space inference.
        return None
    return InputSpec(
        kind=metadata.kind,
        shape=metadata.shape,
        layout=metadata.layout,
        feature_names=metadata.feature_names,
        metadata=metadata.metadata,
    )


def _prediction_summaries(
    *,
    forward_output: torch.Tensor,
    sample_ids: list[str] | None,
    targets: torch.Tensor | None,
) -> tuple[PredictionSummary, ...]:
    if forward_output.ndim != 2 or forward_output.shape[1] < 2:
        return ()

    probabilities = torch.softmax(forward_output.detach().cpu(), dim=1)
    confidences, predictions = probabilities.max(dim=1)
    resolved_targets = _valid_targets_for_reporting(
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


def _valid_targets_for_reporting(
    *,
    targets: torch.Tensor | None,
    expected: int,
) -> torch.Tensor | None:
    if targets is None:
        return None
    if targets.ndim != 1 or int(targets.shape[0]) != expected:
        return None
    return targets.detach().cpu()
