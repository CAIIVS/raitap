from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap.configs import register_configs
from raitap.run.forward_output import extract_primary_tensor
from raitap.run.outputs import PredictionSummary, RunOutputs

if TYPE_CHECKING:
    from raitap.metrics import metrics_prediction_pair, resolve_metric_targets

register_configs()


def main() -> None:
    from raitap.run.__main__ import main as _main

    _main()


def run(*args: Any, **kwargs: Any) -> RunOutputs:
    from raitap.run.pipeline import run as _run

    return _run(*args, **kwargs)


def print_summary(*args: Any, **kwargs: Any) -> None:
    from raitap.run.pipeline import print_summary as _print_summary

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
]
