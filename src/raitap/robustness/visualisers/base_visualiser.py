"""Base class for RAITAP robustness visualisers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from ..contracts import MethodKind
from ..exceptions import MethodKindVisualiserIncompatibilityError

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ..contracts import RobustnessVisualisationContext
    from ..results import RobustnessResult


class BaseRobustnessVisualiser(ABC):
    """All robustness visualisers extend this class.

    Subclasses declare which method kinds they support via the
    ``supported_method_kinds`` ClassVar; the factory enforces compatibility at
    YAML parse time so configuration errors fail fast.
    """

    supported_method_kinds: ClassVar[frozenset[MethodKind]] = frozenset()

    def validate_result(self, result: RobustnessResult) -> None:
        if not self.supported_method_kinds:
            return
        if result.method_kind not in self.supported_method_kinds:
            raise MethodKindVisualiserIncompatibilityError(
                assessor_target=result.assessor_target,
                visualiser=type(self).__name__,
                assessor_method_kind=result.method_kind.value,
                supported_method_kinds=[k.value for k in sorted(self.supported_method_kinds)],
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
