"""Assessment-phase chain — the registry that lists the deliverable phases.

Each phase class lives in its owning module (``MetricsPhase`` in
``metrics/phase.py``, ``TransparencyPhase`` in ``transparency/phase.py``,
``RobustnessPhase`` in ``robustness/phase.py``) next to its work function; the
result type + report rendering sit alongside in ``*/report.py``. This module
only assembles them into :data:`ASSESSMENT_PHASES` — the single list that
the configured-phase guard and the run loop in
:func:`raitap.pipeline.orchestrator.run_without_tracking` iterate. Adding a
module means writing its phase + result in that module, then adding one import +
one entry here.

Shaped after the Chain-of-Responsibility pattern: each phase independently
decides — via :meth:`AssessmentPhase.is_configured` — whether to contribute.
Unlike classic CoR there is no short-circuit: every configured phase runs and
the pipeline accumulates all results, so the chain is a plain ordered list
rather than a ``set_next`` linked list.

``AssessmentPhase`` + ``PhaseContext`` are re-exported from
:mod:`raitap.pipeline.phases.base` so existing importers keep working.
"""

from __future__ import annotations

from raitap.metrics.phase import MetricsPhase
from raitap.pipeline.phases.base import AssessmentPhase, PhaseContext
from raitap.robustness.phase import RobustnessPhase
from raitap.transparency.phase import TransparencyPhase

__all__ = ["ASSESSMENT_PHASES", "AssessmentPhase", "PhaseContext"]

# Ordered chain. Order is preserved in execution (metrics -> transparency ->
# robustness, matching the historical orchestrator order).
ASSESSMENT_PHASES: tuple[AssessmentPhase, ...] = (
    MetricsPhase(),
    TransparencyPhase(),
    RobustnessPhase(),
)
