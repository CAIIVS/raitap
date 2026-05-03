"""Tabular modality visualization (feature importance bars)"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np

from raitap.transparency.contracts import (
    ExplanationOutputSpace,
    ExplanationScope,
    MethodFamily,
    ScopeDefinitionStep,
    VisualSummarySpec,
)

from .base_visualiser import BaseVisualiser

if TYPE_CHECKING:
    import torch
    from matplotlib.figure import Figure

    from raitap.transparency.contracts import VisualisationContext


class TabularBarChartVisualiser(BaseVisualiser):
    """
    Visualise attributions for tabular data as bar charts.

    Works with any attribution method (Captum, SHAP, etc.)
    """

    supported_scopes: ClassVar[frozenset[ExplanationScope]] = frozenset({ExplanationScope.LOCAL})
    supported_output_spaces: ClassVar[frozenset[ExplanationOutputSpace]] = frozenset(
        {
            ExplanationOutputSpace.INPUT_FEATURES,
            ExplanationOutputSpace.INTERPRETABLE_FEATURES,
        }
    )
    supported_method_families: ClassVar[frozenset[MethodFamily]] = frozenset(MethodFamily)
    produces_scope: ClassVar[ExplanationScope | None] = ExplanationScope.COHORT
    scope_definition_step: ClassVar[ScopeDefinitionStep | None] = (
        ScopeDefinitionStep.VISUALISER_SUMMARY
    )
    visual_summary: ClassVar[VisualSummarySpec | None] = VisualSummarySpec(
        summary_type="tabular_bar",
        aggregation="mean_absolute_attribution",
        description="Mean absolute attribution by tabular feature.",
    )

    def __init__(self, feature_names: list[str] | None = None):
        """
        Args:
            feature_names: List of feature names for x-axis labels
        """
        self.feature_names = feature_names

    def validate_explanation(
        self,
        explanation: object,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None,
    ) -> None:
        super().validate_explanation(explanation, attributions, inputs)

        semantics = getattr(explanation, "semantics", None)
        input_spec = getattr(semantics, "input_spec", None)
        output_space = getattr(semantics, "output_space", None)
        kind = str(getattr(input_spec, "kind", "") or "").lower()
        input_layout = str(getattr(input_spec, "layout", "") or "").upper().replace(" ", "")
        layout = str(getattr(output_space, "layout", "") or "").upper().replace(" ", "")

        if kind in {"image", "text", "time_series", "timeseries"}:
            self._raise_incompatibility(
                "input metadata",
                kind,
                "tabular",
            )
        if kind == "tabular" or input_layout in {"B,F", "(B,F)"} or layout in {"B,F", "(B,F)"}:
            return
        self._raise_incompatibility(
            "tabular layout",
            kind or input_layout or layout,
            "(B, F) tabular/interpretable attributions",
        )

    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        *,
        context: VisualisationContext | None = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> Figure:
        """
        Create feature importance bar chart.

        Args:
            attributions: (B, num_features) array
            inputs: Not used for tabular visualization
            context: Standard RAITAP metadata (not used by this aggregate visualiser)
            title: Optional chart title override

        Returns:
            Matplotlib figure
        """
        del context, kwargs

        # Convert to numpy
        if hasattr(attributions, "detach"):
            attrs_np = attributions.detach().cpu().numpy()
        elif hasattr(attributions, "numpy"):
            attrs_np = attributions.cpu().numpy()
        else:
            attrs_np = np.array(attributions)

        # Aggregate across batch (mean absolute attribution)
        mean_importance = np.abs(attrs_np).mean(axis=0)

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(mean_importance))
        ax.bar(x, mean_importance)

        if self.feature_names:
            ax.set_xticks(x)
            ax.set_xticklabels(self.feature_names, rotation=45, ha="right")

        ax.set_ylabel("Mean Absolute Attribution")
        ax.set_xlabel("Features")
        ax.set_title(title or "Feature Importance")
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        return fig
