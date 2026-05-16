"""Force adapter registration + apply Hydra-only specialisations.

Adapter classes register themselves via :class:`raitap._adapters.AdapterMixin`
at class-creation time. That requires the class file to be imported.
Importing each module's ``__init__.py`` here is enough — those packages
re-export their leaf adapters, which triggers ``__init_subclass__``.

A handful of bundled Hydra group entries need shapes the mixin can't
produce (``# @package _global_`` callback injection for reporting multirun,
``_target_: null`` for the ``disabled`` reporting variant). Those live
below as direct :class:`hydra.core.config_store.ConfigStore` writes.
"""

from __future__ import annotations

from dataclasses import field, make_dataclass

from hydra.core.config_store import ConfigStore

from raitap._adapters import store

_HTML_REPORTER_TARGET = "HTMLReporter"
_PDF_REPORTER_TARGET = "PDFReporter"
_REPORTING_SWEEP_CALLBACK_TARGET = "raitap.reporting.hydra_callback.ReportingSweepCallback"


def _callback_block() -> dict[str, dict[str, dict[str, str]]]:
    """Fresh per-entry so downstream mutation cannot leak between groups."""
    return {"callbacks": {"reporting_sweep": {"_target_": _REPORTING_SWEEP_CALLBACK_TARGET}}}


_REGISTERED = False


def register_zen_groups() -> None:
    """Idempotent. Triggers adapter ``__init_subclass__`` for every module,
    then layers the Hydra-only specialisations on top.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    # Importing each subpackage runs its ``__init__.py``, which imports the
    # leaf adapter modules → ``AdapterMixin.__init_subclass__`` fires. These
    # are pure side-effect imports; we discard the module objects through
    # ``importlib`` so both ruff and pyright stay quiet.
    import importlib

    for pkg in (
        "raitap.metrics",
        "raitap.reporting",
        "raitap.robustness",
        "raitap.tracking",
        "raitap.transparency",
    ):
        importlib.import_module(pkg)

    # Flush the mixin-generated entries first; the special-cases below then
    # overwrite the reporting entries with the ``_global_`` + multirun shape.
    store.add_to_hydra_store(overwrite_ok=True)

    cs = ConfigStore.instance()

    # Reporting variants that the mixin can't model:
    # - ``html`` / ``pdf`` need ``# @package _global_`` to inject the multirun
    #   callback alongside the ``reporting`` node.
    # - ``disabled`` carries ``_target_: null`` + ``multirun_report: false``.
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

    from .schema import ReportingConfig

    reporting_disabled_node = make_dataclass(
        "_ReportingDisabledNode",
        [
            ("_target_", str | None, field(default=None)),
            ("multirun_report", bool, field(default=False)),
        ],
        bases=(ReportingConfig,),
    )
    # Direct ConfigStore write (not via hydra-zen store) so it goes straight
    # into the live ConfigStore after the flush above.
    cs.store(group="reporting", name="disabled", node=reporting_disabled_node)
