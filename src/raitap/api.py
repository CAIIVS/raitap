"""Public Python API.

Programmatic counterpart to ``raitap --config-name ...``. Same
:class:`AppConfig`, same orchestrator. No Hydra working-dir chdir, no
logging hijack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hydra_zen import builds, instantiate

from raitap.configs.schema import (
    AppConfig,
    DataConfig,
    LabelsConfig,
    MetricsConfig,
    ModelConfig,
    ReportingConfig,
    RobustnessConfig,
    TrackingConfig,
    TransparencyConfig,
)
from raitap.metrics.classification_metrics import ClassificationMetrics
from raitap.pipeline.orchestrator import run as _orchestrator_run
from raitap.robustness.assessors.foolbox_assessor import FoolboxAssessor
from raitap.robustness.assessors.torchattacks_assessor import TorchattacksAssessor
from raitap.transparency.explainers.captum_explainer import CaptumExplainer
from raitap.transparency.explainers.shap_explainer import ShapExplainer

if TYPE_CHECKING:
    from raitap.pipeline.outputs import RunOutputs

__all__ = [
    "AppConfig",
    "DataConfig",
    "LabelsConfig",
    "MetricsConfig",
    "ModelConfig",
    "ReportingConfig",
    "RobustnessConfig",
    "TrackingConfig",
    "TransparencyConfig",
    "captum",
    "classification_metrics",
    "foolbox",
    "instantiate",
    "run",
    "shap",
    "torchattacks",
]


def run(config: AppConfig, *, verbose: bool = True) -> RunOutputs:
    """Run the full pipeline programmatically.

    No chdir, no logging hijack. Pass ``verbose=False`` to suppress the
    summary panel + report-generation log line.
    """
    return _orchestrator_run(config, verbose=verbose)


captum = builds(CaptumExplainer, populate_full_signature=True, zen_partial=False)
shap = builds(ShapExplainer, populate_full_signature=True)
torchattacks = builds(TorchattacksAssessor, populate_full_signature=True)
foolbox = builds(FoolboxAssessor, populate_full_signature=True)
classification_metrics = builds(ClassificationMetrics, populate_full_signature=True)
