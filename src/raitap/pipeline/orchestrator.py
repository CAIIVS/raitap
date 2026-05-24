"""Pipeline orchestration: ``_run_pipeline`` (full, with tracker context) and
``run_without_tracking`` (composition helper for tests + embedded callers).

The actual phase work lives under :mod:`raitap.pipeline.phases`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raitap import raitap_log
from raitap.data import Data
from raitap.data.preprocessing import resolve_preprocessing
from raitap.models import Model
from raitap.pipeline.outputs import RunOutputs
from raitap.pipeline.phases.assess_robustness import assess_robustness
from raitap.pipeline.phases.assess_transparency import assess_transparency
from raitap.pipeline.phases.evaluate_metrics import evaluate_metrics
from raitap.pipeline.phases.forward_pass import forward_pass
from raitap.pipeline.phases.input_metadata import input_metadata_for_data
from raitap.pipeline.phases.prediction_summaries import prediction_summaries
from raitap.pipeline.ui import print_summary
from raitap.reporting import build_report, create_report, reporting_enabled
from raitap.reporting.sample_selection import resolve_report_sample_selection
from raitap.tracking import BaseTracker
from raitap.utils.lazy import lazy_import

if TYPE_CHECKING:
    import torch

    from raitap.configs.schema import AppConfig
    from raitap.data.preprocessing import ResolvedPreprocessing
else:
    torch = lazy_import("torch")


def _run_pipeline(
    config: AppConfig,
    *,
    verbose: bool = True,
    acknowledge_preprocessing_off: bool = False,
    acknowledge_preprocessing_exec: bool = False,
    allow_unsafe_pickle: bool = False,
) -> RunOutputs:
    """Run the full assessment pipeline, including reporting and tracker logging.

    Parameters
    ----------
    config:
        The fully-resolved application configuration.
    verbose:
        When ``True`` (the default), print the run summary panel and the
        "Generating report..." status line. When ``False``, suppress both —
        leaving phase-level progress logs under standard ``logging``
        control. ``logging`` itself is not reconfigured; programmatic callers
        wanting full silence should raise the root log level.
    acknowledge_preprocessing_off / acknowledge_preprocessing_exec / allow_unsafe_pickle:
        Forwarded to the preprocessing resolver and the :class:`Model`
        loader — see :func:`raitap.run` for the user-facing semantics.
    """
    # Defer warnings emitted during model + data construction so the
    # summary panel renders first; otherwise the rich handler interleaves
    # them above the banner and makes the run header look fragmented.
    with raitap_log.deferred():
        resolved_preprocessing = resolve_preprocessing(
            config.model,
            config.data,
            acknowledge_off=acknowledge_preprocessing_off,
            acknowledge_exec=acknowledge_preprocessing_exec,
        )
        model = Model(
            config,
            resolved_preprocessing=resolved_preprocessing,
            allow_unsafe_pickle=allow_unsafe_pickle,
        )
        data = Data(
            config,
            resolved_preprocessing=resolved_preprocessing,
            task_kind=model.backend.task_kind,
        )
    _validate_report_sample_selection(config, data)
    if verbose:
        print_summary(config, model)

    outputs = run_without_tracking(
        config,
        model,
        data,
        resolved_preprocessing=resolved_preprocessing,
    )

    report_generation = None
    if reporting_enabled(config):
        if verbose:
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
        reporting_cfg = getattr(config, "reporting", None)
        if report_generation is not None and reporting_cfg is not None:
            report_generation.log(tracker)

    return outputs


def run_without_tracking(
    config: AppConfig,
    model: Model,
    data: Data,
    *,
    resolved_preprocessing: ResolvedPreprocessing | None = None,
) -> RunOutputs:
    """Compose every phase except tracker setup.

    Public helper for tests and embedded callers that want the pipeline output
    without instantiating a tracker. Composes the phase modules under
    :mod:`raitap.pipeline.phases` in order.
    """
    raitap_log.info("Running model forward pass...")
    with torch.no_grad():
        forward_output = forward_pass(config, model.backend, data.tensor)

    metrics_eval = evaluate_metrics(config, forward_output, data.labels)

    if not (getattr(config, "transparency", None) or getattr(config, "robustness", None)):
        raise ValueError("No explainers or robustness assessors configured")

    input_metadata = input_metadata_for_data(config, data)
    explanations, visualisations = assess_transparency(
        config,
        model,
        data,
        forward_output,
        input_metadata=input_metadata,
        resolved_preprocessing=resolved_preprocessing,
    )
    robustness_results, robustness_visualisations = assess_robustness(
        config,
        model,
        data,
        forward_output,
        labels=data.labels,
        input_metadata=input_metadata,
        resolved_preprocessing=resolved_preprocessing,
    )

    return RunOutputs(
        explanations=explanations,
        visualisations=visualisations,
        metrics=metrics_eval,
        forward_output=forward_output,
        sample_ids=data.sample_ids,
        targets=data.labels,
        prediction_summaries=prediction_summaries(
            forward_output=forward_output,
            sample_ids=data.sample_ids,
            targets=data.labels,
        ),
        robustness_results=robustness_results,
        robustness_visualisations=robustness_visualisations,
    )


def _validate_report_sample_selection(config: AppConfig, data: Data) -> None:
    if not reporting_enabled(config):
        return
    reporting_cfg = config.reporting
    selection = None if reporting_cfg is None else getattr(reporting_cfg, "sample_selection", None)
    resolve_report_sample_selection(
        selection,
        sample_ids=data.sample_ids,
        batch_size=len(data.tensor),
    )
