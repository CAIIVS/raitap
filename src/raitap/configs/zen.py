"""Hydra-zen ``store`` registrations replacing the per-group YAML files.

These calls reproduce the (group, name) entries that previously lived as
``src/raitap/configs/<group>/<name>.yaml`` files. Importing this module and
calling :func:`register_zen_groups` makes the entries available through
Hydra's ``ConfigStore`` so ``--config-name demo <group>=<name>`` overrides
continue to work.

Design notes
------------
The previous YAML files were target-only (just ``_target_: <name>``) so the
schema dataclasses (``TransparencyConfig`` / ``RobustnessConfig`` / ...) kept
their default field values until ``demo.yaml`` or a user config inlined the
real kwargs. Using ``populate_full_signature=True`` would diverge from that
behaviour by injecting ``algorithm: ???`` etc., and (because the resulting
hydra-zen dataclass does not subclass the schema config) breaks the
structured-config merge that ``raitap_schema`` performs.

The fix is twofold:

1. Skip ``populate_full_signature`` for everything that flows into a typed
   schema slot (transparency / robustness / metrics / tracking / reporting):
   ``builds(Cls)`` alone sets ``_target_`` and nothing else, matching the old
   YAML.
2. Subclass the schema dataclass via ``builds_bases=(SchemaCls,)`` so the
   stored node satisfies the OmegaConf merge rules under the schema-typed
   parent (e.g. ``dict[str, TransparencyConfig]``).
"""

from __future__ import annotations

from typing import Any

from hydra.core.config_store import ConfigStore
from hydra_zen import builds, store

from raitap.metrics.classification_metrics import ClassificationMetrics
from raitap.metrics.detection_metrics import DetectionMetrics
from raitap.robustness.assessors.foolbox_assessor import FoolboxAssessor
from raitap.robustness.assessors.marabou_assessor import MarabouAssessor
from raitap.robustness.assessors.torchattacks_assessor import TorchattacksAssessor
from raitap.transparency.explainers.captum_explainer import CaptumExplainer
from raitap.transparency.explainers.shap_explainer import ShapExplainer

from .schema import (
    MetricsConfig,
    ReportingConfig,
    RobustnessConfig,
    TrackingConfig,
    TransparencyConfig,
)

# Short-name ``_target_`` strings (resolved at runtime by
# :func:`raitap.configs.utils.resolve_target`), matching the old YAML files
# which used unqualified class names (e.g. ``_target_: HTMLReporter``).
_HTML_REPORTER_TARGET = "HTMLReporter"
_PDF_REPORTER_TARGET = "PDFReporter"
_MLFLOW_TRACKER_TARGET = "MLFlowTracker"
# The Hydra callback target is fully-qualified because ``resolve_target`` is
# not invoked on the ``hydra.callbacks`` block — Hydra instantiates it
# directly and needs an importable dotted path.
_REPORTING_SWEEP_CALLBACK_TARGET = "raitap.reporting.hydra_callback.ReportingSweepCallback"


def register_zen_groups() -> None:
    """Idempotent: register every group/name pair into Hydra's ConfigStore.

    Matches the previous YAML layout under ``src/raitap/configs/<group>/``.
    Reporting entries also inject the multirun ``reporting_sweep`` callback
    via ``package="_global_"`` to mirror the old ``# @package _global_``
    semantics of ``reporting/html.yaml`` and ``reporting/pdf.yaml``.
    """
    # --- transparency -----------------------------------------------------
    # Old YAMLs used ``# @package transparency.<name>`` so the schema dict
    # field ``transparency: dict[str, TransparencyConfig]`` gains a sibling
    # entry rather than being replaced wholesale.
    store(
        builds(CaptumExplainer, builds_bases=(TransparencyConfig,)),
        group="transparency",
        name="captum",
        package="transparency.captum",
    )
    store(
        builds(ShapExplainer, builds_bases=(TransparencyConfig,)),
        group="transparency",
        name="shap",
        package="transparency.shap",
    )

    # --- robustness -------------------------------------------------------
    store(
        builds(TorchattacksAssessor, builds_bases=(RobustnessConfig,)),
        group="robustness",
        name="torchattacks",
        package="robustness.torchattacks",
    )
    store(
        builds(FoolboxAssessor, builds_bases=(RobustnessConfig,)),
        group="robustness",
        name="foolbox",
        package="robustness.foolbox",
    )
    store(
        builds(MarabouAssessor, builds_bases=(RobustnessConfig,)),
        group="robustness",
        name="marabou",
        package="robustness.marabou",
    )

    # --- metrics ----------------------------------------------------------
    # ``classification.yaml`` set ``task: multiclass`` explicitly.
    store(
        builds(ClassificationMetrics, task="multiclass", builds_bases=(MetricsConfig,)),
        group="metrics",
        name="classification",
    )
    store(
        builds(DetectionMetrics, builds_bases=(MetricsConfig,)),
        group="metrics",
        name="detection",
    )

    # --- reporting --------------------------------------------------------
    # html / pdf use ``# @package _global_`` so they inject both the
    # ``reporting`` node and the multirun callback into the root config. The
    # node has to be a plain dict (not a structured dataclass) so the global
    # merge does not run AppConfig's subclass check — matching the old YAML
    # behaviour, which produced a free-form DictConfig at composition time.
    def _callback_block() -> dict[str, Any]:
        # Fresh dict per store entry so downstream Hydra mutation of one
        # group's callback config cannot leak into siblings.
        return {"callbacks": {"reporting_sweep": {"_target_": _REPORTING_SWEEP_CALLBACK_TARGET}}}

    cs = ConfigStore.instance()
    cs.store(
        group="reporting",
        name="html",
        package="_global_",
        node={
            "reporting": {"_target_": _HTML_REPORTER_TARGET},
            "hydra": _callback_block(),
        },
    )
    cs.store(
        group="reporting",
        name="pdf",
        package="_global_",
        node={
            "reporting": {"_target_": _PDF_REPORTER_TARGET},
            "hydra": _callback_block(),
        },
    )
    # ``disabled`` mirrors the old ``_target_: null`` + ``multirun_report: false``.
    # ``make_config`` reserves the ``_target_`` field name, so we construct a
    # plain dataclass subclass via ``dataclasses.make_dataclass``.
    from dataclasses import field, make_dataclass

    reporting_disabled_node = make_dataclass(
        "_ReportingDisabledNode",
        [
            ("_target_", str | None, field(default=None)),
            ("multirun_report", bool, field(default=False)),
        ],
        bases=(ReportingConfig,),
    )
    store(reporting_disabled_node, group="reporting", name="disabled", to_config=lambda x: x)

    # --- tracking ---------------------------------------------------------
    mlflow_tracking_node = make_dataclass(
        "_MLFlowTrackingNode",
        [("_target_", str, field(default=_MLFLOW_TRACKER_TARGET))],
        bases=(TrackingConfig,),
    )
    store(mlflow_tracking_node, group="tracking", name="mlflow", to_config=lambda x: x)

    # Flush all pending entries into Hydra's global ConfigStore. ``store``
    # records additions lazily; this call materialises them so Hydra sees
    # the groups during composition.
    store.add_to_hydra_store(overwrite_ok=True)
