"""Base class for RAITAP robustness visualisers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from raitap._adapters import AdapterMixin

from ..exceptions import AssessmentKindVisualiserIncompatibilityError

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ..contracts import AssessmentKind, RobustnessVisualisationContext
    from ..results import RobustnessResult


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
    """

    supported_assessment_kinds: ClassVar[frozenset[AssessmentKind]] = frozenset()
    # Class-level because embedding robustness facets is a visual layout property.
    embeds_clean_input: ClassVar[bool] = False
    embeds_perturbation_map: ClassVar[bool] = False

    def validate_result(self, result: RobustnessResult) -> None:
        if not self.supported_assessment_kinds:
            return
        if result.assessment_kind not in self.supported_assessment_kinds:
            raise AssessmentKindVisualiserIncompatibilityError(
                assessor_target=result.assessor_target,
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
