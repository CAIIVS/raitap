"""Pipeline orchestration: ``_run_pipeline`` (full, with tracker context) and
``run_without_tracking`` (composition helper for tests + embedded callers).

The actual phase work lives under :mod:`raitap.pipeline.phases`."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from raitap import raitap_log
from raitap.configs import resolve_run_dir
from raitap.data import Data
from raitap.data.preprocessing import resolve_preprocessing
from raitap.models import Model
from raitap.pipeline.outputs import RunOutputs
from raitap.pipeline.phases.forward_pass import forward_pass
from raitap.pipeline.phases.input_metadata import input_metadata_for_data
from raitap.pipeline.phases.prediction_summaries import prediction_summaries
from raitap.pipeline.phases.registry import ASSESSMENT_PHASES, PhaseContext
from raitap.pipeline.ui import print_summary
from raitap.reporting import build_report, create_report, reporting_enabled
from raitap.reporting.sample_selection import resolve_report_sample_selection
from raitap.reproducibility import (
    assess_reproducibility,
    pin_global_seed,
    reproducibility_caveat,
    write_reproducibility_md,
)
from raitap.tracking import BaseTracker
from raitap.utils.diagnostics import Module
from raitap.utils.lazy import lazy_import

if TYPE_CHECKING:
    import torch

    from raitap.configs.schema import AppConfig
    from raitap.data.preprocessing import ResolvedPreprocessing
    from raitap.pipeline.outputs import PhaseResult
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
        # Render the summary panel inside the deferred block: it prints
        # immediately (direct rich console), so the deferred construction logs
        # ("Preprocessing: …") + any setup warnings replay *after* the panel.
        if verbose:
            print_summary(config, model)

    seed = getattr(config, "seed", None)
    if seed is not None:
        pin_global_seed(seed)
    outputs = run_without_tracking(
        config,
        model,
        data,
        resolved_preprocessing=resolved_preprocessing,
    )

    # Reproducibility caveat (#251, #339). Derived once from the run's semantics
    # and the (maybe-unset) seed. The output-dir note and the warning fire
    # whenever there is something to warn about — independent of reporting (the
    # run dir + console exist regardless); only the report banner (inside
    # build_report) is gated on reporting.
    repro = assess_reproducibility(outputs, getattr(config, "seed", None))
    # Write the run-level reproducibility artefact when there is something to
    # warn about OR a seed was pinned (record the seed run-wide even if fully
    # reproducible). Not duplicated into per-module metadata.
    if repro.warned or repro.seed is not None:
        write_reproducibility_md(resolve_run_dir(config), repro)

    report_generation = None
    if reporting_enabled(config):
        if verbose:
            # Logged from the orchestrator but logically a reporting concern.
            raitap_log.info("Generating report...", module=Module.reporting)
        report = build_report(config, outputs)
        report_generation = create_report(config=config, report=report)

    if repro.warned:
        # After the "Report generated" log so it reads as a closing caveat.
        # ``reproducibility_caveat`` only returns ``None`` when nothing is
        # warned, which the guard above rules out.
        raitap_log.warn(cast("str", reproducibility_caveat(repro)))

    tracking_config = getattr(config, "tracking", None)
    has_tracker = bool(tracking_config and getattr(tracking_config, "_target_", None))
    if not has_tracker:
        return outputs

    with BaseTracker.create_tracker(config) as tracker:
        tracker.log_config()
        if getattr(config.tracking, "log_model", False):
            model.log(tracker)
        data.log(tracker)
        # Each phase result owns how it logs itself (artifacts + subdirectories).
        for phase_result in outputs.phase_results.values():
            phase_result.log(tracker)
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
    """Compose every assessment phase except tracker setup.

    Public helper for tests and embedded callers that want the pipeline output
    without instantiating a tracker. Resolves which phases are configured from
    :data:`~raitap.pipeline.phases.registry.ASSESSMENT_PHASES`, then runs each
    over a shared :class:`~raitap.pipeline.phases.registry.PhaseContext`.
    """
    # Config-only guard, before the (potentially expensive) forward pass: a run
    # with no configured deliverable phase fails fast instead of after the fact.
    configured_phases = [phase for phase in ASSESSMENT_PHASES if phase.is_configured(config)]
    if not configured_phases:
        _raise_no_phase_configured()

    # Logged from the orchestrator but logically a model operation (same pattern
    # as the reporting log below) so the chip reads "Models", not blank/infra.
    raitap_log.info("Running model forward pass...", module=Module.models)
    with torch.no_grad():
        forward_output = forward_pass(config, model.backend, data.tensor)

    context = PhaseContext(
        config=config,
        model=model,
        data=data,
        forward_output=forward_output,
        input_metadata=input_metadata_for_data(config, data),
        resolved_preprocessing=resolved_preprocessing,
    )
    phase_results: dict[str, PhaseResult] = {}
    for phase in configured_phases:
        result = phase.run(context)
        if result is not None:
            phase_results[phase.name] = result

    return RunOutputs(
        forward_output=forward_output,
        phase_results=phase_results,
        sample_ids=data.sample_ids,
        targets=data.labels,
        prediction_summaries=prediction_summaries(
            forward_output=forward_output,
            sample_ids=data.sample_ids,
            targets=data.labels,
        ),
    )


def _raise_no_phase_configured() -> None:
    available = ", ".join(phase.name for phase in ASSESSMENT_PHASES)
    raise ValueError(f"No assessment phase configured; configure at least one of: {available}")


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
