"""Tests for visualiser compatible_algorithms class attributes"""

from __future__ import annotations

from raitap.transparency.visualisers import (
    CaptumImageVisualiser,
    CaptumTextVisualiser,
    CaptumTimeSeriesVisualiser,
    ShapBarVisualiser,
    ShapBeeswarmVisualiser,
    ShapForceVisualiser,
    ShapImageVisualiser,
    ShapWaterfallVisualiser,
    TabularBarChartVisualiser,
)
from raitap.transparency.visualisers.base import BaseVisualiser


class TestCompatibleAlgorithmsAttribute:
    """Assert that every visualiser honours the compatible_algorithms contract."""

    def test_base_visualiser_default_is_empty_frozenset(self):
        """BaseVisualiser.compatible_algorithms defaults to frozenset() (= unrestricted)."""
        assert BaseVisualiser.compatible_algorithms == frozenset()

    # ------------------------------------------------------------------
    # Captum visualisers: compatible with ALL algorithms (empty frozenset)
    # ------------------------------------------------------------------

    def test_captum_image_unrestricted(self):
        assert CaptumImageVisualiser.compatible_algorithms == frozenset()

    def test_captum_time_series_unrestricted(self):
        assert CaptumTimeSeriesVisualiser.compatible_algorithms == frozenset()

    def test_captum_text_unrestricted(self):
        assert CaptumTextVisualiser.compatible_algorithms == frozenset()

    # ------------------------------------------------------------------
    # SHAP tabular/generic visualisers: unrestricted
    # ------------------------------------------------------------------

    def test_shap_bar_unrestricted(self):
        assert ShapBarVisualiser.compatible_algorithms == frozenset()

    def test_shap_beeswarm_unrestricted(self):
        assert ShapBeeswarmVisualiser.compatible_algorithms == frozenset()

    def test_shap_waterfall_unrestricted(self):
        assert ShapWaterfallVisualiser.compatible_algorithms == frozenset()

    def test_shap_force_unrestricted(self):
        assert ShapForceVisualiser.compatible_algorithms == frozenset()

    # ------------------------------------------------------------------
    # ShapImageVisualiser: RESTRICTED to gradient-based explainers only
    # ------------------------------------------------------------------

    def test_shap_image_restricted_to_gradient_based(self):
        assert ShapImageVisualiser.compatible_algorithms == frozenset(
            {"GradientExplainer", "DeepExplainer"}
        )

    def test_shap_image_accepts_gradient_explainer(self):
        assert "GradientExplainer" in ShapImageVisualiser.compatible_algorithms

    def test_shap_image_accepts_deep_explainer(self):
        assert "DeepExplainer" in ShapImageVisualiser.compatible_algorithms

    def test_shap_image_rejects_kernel_explainer(self):
        assert "KernelExplainer" not in ShapImageVisualiser.compatible_algorithms

    def test_shap_image_rejects_tree_explainer(self):
        assert "TreeExplainer" not in ShapImageVisualiser.compatible_algorithms

    # ------------------------------------------------------------------
    # Framework-agnostic visualisers: unrestricted
    # ------------------------------------------------------------------

    def test_tabular_bar_chart_unrestricted(self):
        assert TabularBarChartVisualiser.compatible_algorithms == frozenset()

    # ------------------------------------------------------------------
    # Type check: attribute must be a frozenset on every concrete class
    # ------------------------------------------------------------------

    def test_all_visualisers_have_frozenset_attribute(self):
        classes = [
            CaptumImageVisualiser,
            CaptumTextVisualiser,
            CaptumTimeSeriesVisualiser,
            ShapBarVisualiser,
            ShapBeeswarmVisualiser,
            ShapWaterfallVisualiser,
            ShapForceVisualiser,
            ShapImageVisualiser,
            TabularBarChartVisualiser,
        ]
        for cls in classes:
            assert isinstance(cls.compatible_algorithms, frozenset), (
                f"{cls.__name__}.compatible_algorithms must be a frozenset"
            )
