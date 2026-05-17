"""Top-level public surface for raitap.

Orchestration-level imports only. Schema dataclasses, builders, and
module-local enums live next to their owning module so the unit of
ownership is consistent across both type and value::

    from raitap import AppConfig, run, Hardware           # orchestration
    from raitap.models import ModelConfig
    from raitap.data import DataConfig, LabelsConfig, LabelEncoding, IdStrategy
    from raitap.metrics import MetricsConfig, Task, classification
    from raitap.transparency import TransparencyConfig, captum, captum_image
    from raitap.robustness import RobustnessConfig, torchattacks, image_pair
    from raitap.reporting import ReportingConfig, html
    from raitap.tracking import TrackingConfig, mlflow

``AppConfig`` is the only schema dataclass at this level because it is the
root the orchestrator consumes; ``Hardware`` is similarly cross-cutting
(orchestrator + deps). Everything else is module-owned.

Names are resolved lazily via :pep:`562` ``__getattr__`` so ``import raitap``
stays cheap (no torchmetrics / Captum / torchattacks import on a bare
``pip install raitap``). The ``TYPE_CHECKING`` block lists names statically
so editor autocomplete still surfaces them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap.utils.log import raitap_log

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig
    from raitap.pipeline.outputs import RunOutputs
    from raitap.types import Hardware

    def run(
        config: AppConfig,
        *,
        verbose: bool = True,
        output_root: str | None = None,
        auto_install_deps: bool = False,
        exec_global: bool = False,
        acknowledge_preprocessing_off: bool = False,
        acknowledge_preprocessing_exec: bool = False,
    ) -> RunOutputs: ...


__all__ = [
    "AppConfig",
    "Hardware",
    "raitap_log",
    "run",
]


_LAZY: dict[str, tuple[str, str]] = {
    "run": ("raitap.api", "run"),
    "AppConfig": ("raitap.configs.schema", "AppConfig"),
    "Hardware": ("raitap.types", "Hardware"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'raitap' has no attribute {name!r}")
    import importlib

    module_path, attr = target
    return getattr(importlib.import_module(module_path), attr)
