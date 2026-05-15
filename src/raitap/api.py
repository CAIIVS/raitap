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


# Builders inherit the schema dataclass via ``builds_bases`` so users can pass
# every field on ``TransparencyConfig`` / ``RobustnessConfig`` / ``MetricsConfig``
# (``constructor``, ``call``, ``raitap``, ``visualisers``) — not just the
# underlying ``__init__`` signature. ``populate_full_signature`` is enabled only
# where the wrapped class exposes typed kwargs (``ClassificationMetrics``);
# explainer / assessor ``__init__``s take ``**kwargs`` which hydra-zen cannot
# introspect, so we rely on the schema base for those.
captum = builds(CaptumExplainer, builds_bases=(TransparencyConfig,))
shap = builds(ShapExplainer, builds_bases=(TransparencyConfig,))
torchattacks = builds(TorchattacksAssessor, builds_bases=(RobustnessConfig,))
foolbox = builds(FoolboxAssessor, builds_bases=(RobustnessConfig,))
classification_metrics = builds(
    ClassificationMetrics,
    builds_bases=(MetricsConfig,),
    populate_full_signature=True,
)
