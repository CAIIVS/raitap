"""Public Python API.

Programmatic counterpart to ``raitap --config-name ...``. Same
:class:`AppConfig`, same orchestrator. No Hydra working-dir chdir, no
logging hijack — but :func:`run` sets a sensible default ``output_root``
mirroring Hydra's layout so artefacts land under ``outputs/<date>/<time>/``
instead of polluting the caller's cwd.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from hydra_zen import builds, instantiate

from raitap.configs import set_output_root
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


def run(
    config: AppConfig,
    *,
    verbose: bool = True,
    output_root: str | Path | None = None,
) -> RunOutputs:
    """Run the full pipeline programmatically.

    No chdir, no logging hijack. Pass ``verbose=False`` to suppress the
    summary panel + report-generation log line.

    Outputs (metrics, reports, transparency artefacts) are written under
    ``output_root``. When ``None``, mirrors Hydra's default and writes to
    ``./outputs/<YYYY-MM-DD>/<HH-MM-SS>/`` so the caller's cwd stays clean.
    Pass an explicit path to redirect, or pre-populate ``config._output_root``
    via :func:`raitap.configs.set_output_root` if you need finer control.
    """
    # OmegaConf ``DictConfig`` raises on missing keys, so guard with try/except
    # instead of ``getattr(..., default)``. Dataclass ``AppConfig`` returns
    # ``None`` naturally; only Hydra-composed configs hit the exception path.
    try:
        existing_root = config._output_root  # type: ignore[attr-defined]
    except Exception:
        existing_root = None
    if output_root is None and existing_root is None:
        now = datetime.now()
        output_root = Path("outputs") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
    if output_root is not None:
        set_output_root(config, output_root)
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
