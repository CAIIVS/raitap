"""Assessment-phase base types.

Split out from ``registry.py`` so each module can define its own
:class:`AssessmentPhase` (importing only this base) without a cycle:
``base`` ← module phase classes ← ``registry`` (which assembles the list).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypeVar

from raitap import raitap_log

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from raitap.configs.schema import AppConfig
    from raitap.data import Data
    from raitap.data.preprocessing import ResolvedPreprocessing
    from raitap.models import Model
    from raitap.pipeline.outputs import ForwardOutput, PhaseResult, _RenderableResult
    from raitap.transparency.contracts import InputSpec


# Bound to the AdapterResult envelope (+ ``_visualise``): pyright proves every
# phase's result satisfies the contract — including the thesis-critical
# ``semantics`` — at the ``run_adapters`` call site.
ResultT = TypeVar("ResultT", bound="_RenderableResult")


def run_adapters(
    names: Iterable[str],
    *,
    log_label: str,
    build_one: Callable[[str], ResultT | None],
) -> list[ResultT]:
    """Run the adapter loop shared by adapter-style phases (transparency, robustness, …).

    For each configured adapter ``name``, ``build_one(name)`` instantiates the
    module's result (or returns ``None`` to skip it). This helper then calls the
    result's ``_visualise()`` — so every result owns its visualisations (issue
    #243) and a contributor cannot forget that contract — and collects the
    results in ``names`` order. Metrics is a singleton (no adapter loop) and does
    not use this helper.
    """
    names = list(names)
    if not names:
        return []
    suffix = "s" if len(names) > 1 else ""
    # ``module=log_label`` so the chip reads e.g. "Robustness" even though this
    # shared loop lives under ``pipeline/phases/`` — the logger name alone would
    # mis-classify it as "pipeline".
    raitap_log.info(
        "Performing %s assessment%s (%d)...", log_label, suffix, len(names), module=log_label
    )
    results: list[ResultT] = []
    for name in names:
        result = build_one(name)
        if result is None:
            continue
        result._visualise()  # populates result.visualisations
        results.append(result)
    return results


@dataclass(frozen=True)
class PhaseContext:
    """Everything an assessment phase may need, assembled once per run."""

    config: AppConfig
    model: Model
    data: Data
    forward_output: ForwardOutput
    input_metadata: InputSpec | None
    resolved_preprocessing: ResolvedPreprocessing | None


class AssessmentPhase(ABC):
    """One deliverable-producing phase: a configured-check plus a run step.

    Concrete phases live in their owning module (e.g. ``TransparencyPhase`` in
    ``transparency/report.py``) and are registered in
    :data:`raitap.pipeline.phases.registry.ASSESSMENT_PHASES`.
    """

    name: ClassVar[str]

    @abstractmethod
    def is_configured(self, config: AppConfig) -> bool:
        """True when this phase has anything to do for ``config`` (config-only)."""

    @abstractmethod
    def run(self, ctx: PhaseContext) -> PhaseResult | None:
        """Run the phase; return its result, or ``None`` when it produced nothing."""
