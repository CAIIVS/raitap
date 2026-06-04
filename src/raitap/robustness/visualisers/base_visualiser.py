"""Base class for RAITAP robustness visualisers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

# Runtime imports: these names appear in public annotations on this base class, so
# typing.get_type_hints() must resolve them from module globals.
from matplotlib.figure import Figure  # noqa: TC002

from raitap._adapters import AdapterMixin

from ..contracts import (
    AssessmentKind,
    ReportFigureScope,
    RobustnessVisualisationContext,
)
from ..exceptions import AssessmentKindVisualiserIncompatibilityError

if TYPE_CHECKING:
    from ..results import RobustnessResult
else:
    # Runtime alias: a real import would be circular (``results`` imports this
    # module for ``BaseRobustnessVisualiser``). ``Any`` lets get_type_hints()
    # resolve the string annotation without the cycle.
    RobustnessResult = Any


class BaseRobustnessVisualiser(ABC, AdapterMixin):
    """All robustness visualisers extend this class.

    Subclasses declare which assessment kinds they support via the
    ``supported_assessment_kinds`` ClassVar; the factory enforces compatibility at
    YAML parse time so configuration errors fail fast.

    Empirical visualisers may also declare class-level facet hints used by
    compact report rendering. Setting ``embeds_clean_input`` or
    ``embeds_perturbation_map`` to ``True`` means the visualiser must accept the
    matching runtime kwarg (``include_clean_input`` /
    ``include_perturbation_map``) and hide that facet when it is ``False``.
    The flags default to ``False``, so formal verifier visualisers are
    unaffected unless they explicitly opt into the contract.

    ``report_figure_scope`` declares whether the produced figure summarises the
    whole assessment (``ASSESSOR`` — one chart per assessor) or shows a single
    input (``PER_SAMPLE``). The reporting layer reads it to choose the layout
    slot; it defaults to ``PER_SAMPLE`` so empirical image visualisers keep their
    existing per-sample placement.
    """

    supported_assessment_kinds: ClassVar[frozenset[AssessmentKind]] = frozenset()
    # Class-level because embedding robustness facets is a visual layout property.
    embeds_clean_input: ClassVar[bool] = False
    embeds_perturbation_map: ClassVar[bool] = False
    # Class-level because figure scope is a visual layout property of the visualiser.
    report_figure_scope: ClassVar[ReportFigureScope] = ReportFigureScope.PER_SAMPLE

    def validate_result(self, result: RobustnessResult) -> None:
        if not self.supported_assessment_kinds:
            return
        if result.assessment_kind not in self.supported_assessment_kinds:
            raise AssessmentKindVisualiserIncompatibilityError(
                assessor_target=result.adapter_target,
                visualiser=type(self).__name__,
                assessor_assessment_kind=result.assessment_kind.value,
                supported_assessment_kinds=[
                    k.value for k in sorted(self.supported_assessment_kinds)
                ],
            )

    @abstractmethod
    def visualise(
        self,
        result: RobustnessResult,
        *,
        context: RobustnessVisualisationContext,
        **kwargs: Any,
    ) -> Figure:
        """Render a figure for ``result``."""
