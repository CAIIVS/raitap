"""Public surface of the ``raitap.pipeline`` package.

Orchestration lives in :mod:`raitap.pipeline.orchestrator`; phase work in
:mod:`raitap.pipeline.phases.*`. This module re-exports the small public API
that callers (tests, downstream consumers, the CLI entry) rely on.

Function imports are intentionally lazy to avoid a chain of eager submodule
loads (orchestrator → reporting → metrics → …) at first ``import raitap.pipeline``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap.configs import register_configs
from raitap.pipeline.outputs import PredictionSummary, RunOutputs
from raitap.pipeline.phases.forward import extract_primary_tensor

if TYPE_CHECKING:
    from raitap.metrics import metrics_prediction_pair, resolve_metric_targets

register_configs()


def main() -> None:
    from raitap.pipeline.__main__ import main as _main

    _main()


def run(*args: Any, **kwargs: Any) -> RunOutputs:
    from raitap.pipeline.orchestrator import run as _run

    return _run(*args, **kwargs)


def run_without_tracking(*args: Any, **kwargs: Any) -> RunOutputs:
    from raitap.pipeline.orchestrator import run_without_tracking as _run_without_tracking

    return _run_without_tracking(*args, **kwargs)


def print_summary(*args: Any, **kwargs: Any) -> None:
    from raitap.pipeline.ui import print_summary as _print_summary

    _print_summary(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name == "metrics_prediction_pair":
        from raitap.metrics import metrics_prediction_pair

        return metrics_prediction_pair
    if name == "resolve_metric_targets":
        from raitap.metrics import resolve_metric_targets

        return resolve_metric_targets
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PredictionSummary",
    "RunOutputs",
    "extract_primary_tensor",
    "main",
    "metrics_prediction_pair",
    "print_summary",
    "resolve_metric_targets",
    "run",
    "run_without_tracking",
]
