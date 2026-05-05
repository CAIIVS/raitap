"""Base class for visualization (modality-agnostic interface)"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import matplotlib.pyplot as plt

from raitap.transparency.contracts import (
    ExplanationOutputSpace,
    ExplanationPayloadKind,
    ExplanationScope,
    MethodFamily,
    ScopeDefinitionStep,
    VisualisationContext,
    VisualSummarySpec,
)

if TYPE_CHECKING:
    import torch
    from matplotlib.figure import Figure


class BaseVisualiser(ABC):
    """
    Abstract base class for all visualiser implementations.

    visualisers ONLY handle visualization - they do NOT compute attributions.
    Attribution computation is handled by separate Explainer classes.

    Class attributes
    ----------------
    compatible_algorithms:
        Frozenset of algorithm names this visualiser supports.
        An empty frozenset (the default) means *compatible with all algorithms*.
    supported_payload_kinds:
        Frozenset of :class:`~raitap.transparency.contracts.ExplanationPayloadKind` values this
        visualiser can render. An **empty** frozenset means *compatible with all payload kinds*
        (wildcard). A non-empty frozenset requires the explainer's ``output_payload_kind`` to be
        included.
    supported_scopes, supported_output_spaces, supported_method_families:
        Typed semantic compatibility declarations. Empty frozensets are wildcards for custom and
        legacy subclasses.
    """

    compatible_algorithms: ClassVar[frozenset[str]] = frozenset()
    supported_payload_kinds: ClassVar[frozenset[ExplanationPayloadKind]] = frozenset(
        {ExplanationPayloadKind.ATTRIBUTIONS}
    )
    supported_scopes: ClassVar[frozenset[ExplanationScope]] = frozenset()
    supported_output_spaces: ClassVar[frozenset[ExplanationOutputSpace]] = frozenset()
    supported_method_families: ClassVar[frozenset[MethodFamily]] = frozenset()
    produces_scope: ClassVar[ExplanationScope | None] = None
    scope_definition_step: ClassVar[ScopeDefinitionStep | None] = None
    visual_summary: ClassVar[VisualSummarySpec | None] = None

    def validate_explanation(
        self,
        explanation: object,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None,
    ) -> None:
        """Validate that this visualiser can render an explanation's typed semantics."""
        del attributions, inputs

        semantics = getattr(explanation, "semantics", None)
        if semantics is None:
            self._raise_incompatibility("semantics", "missing", "typed explanation semantics")

        payload_kind = getattr(semantics, "payload_kind", None)
        if self.supported_payload_kinds and payload_kind not in self.supported_payload_kinds:
            self._raise_incompatibility(
                "payload kind",
                _semantic_value(payload_kind),
                _format_supported(self.supported_payload_kinds),
            )

        scope = getattr(semantics, "scope", None)
        if self.supported_scopes and scope not in self.supported_scopes:
            self._raise_incompatibility(
                "scope",
                _semantic_value(scope),
                _format_supported(self.supported_scopes),
            )

        output_space_spec = getattr(semantics, "output_space", None)
        output_space = getattr(output_space_spec, "space", None)
        if self.supported_output_spaces and output_space not in self.supported_output_spaces:
            self._raise_incompatibility(
                "output space",
                _semantic_value(output_space),
                _format_supported(self.supported_output_spaces),
            )

        method_families = getattr(semantics, "method_families", frozenset())
        if self.supported_method_families and not (
            frozenset(method_families) & self.supported_method_families
        ):
            self._raise_incompatibility(
                "method family",
                _format_supported(method_families),
                _format_supported(self.supported_method_families),
            )

    def _raise_incompatibility(self, dimension: str, actual: str, expected: str) -> None:
        raise ValueError(
            f"Visualiser {type(self).__name__!r} is incompatible with this explanation: "
            f"{dimension} {actual!r} is not supported; expected {expected}."
        )

    @abstractmethod
    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        *,
        context: VisualisationContext | None = None,
        **kwargs: Any,
    ) -> Figure:
        """
        Create visualization from attributions.

        Args:
            attributions: Attribution values (numpy array or tensor)
            inputs: Original inputs for overlay (optional)
            context: Standard RAITAP pipeline metadata (optional)
            **kwargs: visualiser-specific arguments

        Returns:
            Matplotlib figure
        """
        pass

    def save(
        self,
        attributions: torch.Tensor,
        output_path: str | Path,
        inputs: torch.Tensor | None = None,
        *,
        context: VisualisationContext | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Save visualization to file.

        Default implementation using visualise(). Subclasses can override for efficiency.

        Args:
            attributions: Attribution values
            output_path: Path to save image
            inputs: Original inputs for overlay (optional)
            context: Standard RAITAP pipeline metadata (optional)
            **kwargs: visualiser-specific arguments
        """
        fig = self.visualise(attributions, inputs, context=context, **kwargs)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


def _format_supported(values: object) -> str:
    try:
        items = sorted(_semantic_value(value) for value in values)  # type: ignore[union-attr]
    except TypeError:
        return _semantic_value(values)
    return ", ".join(items) if items else "any"


def _semantic_value(value: object) -> str:
    return str(getattr(value, "value", value))
