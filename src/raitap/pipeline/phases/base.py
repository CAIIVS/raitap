"""Assessment-phase base types.

Split out from ``registry.py`` so each module can define its own
:class:`AssessmentPhase` (importing only this base) without a cycle:
``base`` ← module phase classes ← ``registry`` (which assembles the list).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig
    from raitap.data import Data
    from raitap.data.preprocessing import ResolvedPreprocessing
    from raitap.models import Model
    from raitap.pipeline.outputs import ForwardOutput, PhaseResult
    from raitap.transparency.contracts import InputSpec


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
