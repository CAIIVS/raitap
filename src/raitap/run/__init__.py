from __future__ import annotations

from raitap.configs import register_configs
from raitap.run.__main__ import main
from raitap.run.forward_output import extract_primary_tensor
from raitap.run.metrics_placeholder import metrics_prediction_pair
from raitap.run.outputs import RunOutputs
from raitap.run.pipeline import print_summary, run

register_configs()

__all__ = [
    "RunOutputs",
    "extract_primary_tensor",
    "main",
    "metrics_prediction_pair",
    "print_summary",
    "run",
]
