"""Tests for visualiser implementations"""

from __future__ import annotations

import numpy as np
import pytest
import torch

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


class TestCaptumImageVisualiser:
    """Test CaptumImageVisualiser"""

    def test_initialization(self):
        visualiser = CaptumImageVisualiser()
        assert visualiser is not None

    def test_visualise_tensor(self, sample_images):
        visualiser = CaptumImageVisualiser(method="heat_map")
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(attributions)
        assert fig is not None
        assert len(fig.axes) >= 4  # at least one axes per sample

    def test_visualise_numpy(self, sample_images):
        visualiser = CaptumImageVisualiser(method="heat_map")
        attributions = np.random.randn(*sample_images.shape)

        fig = visualiser.visualise(attributions)
        assert fig is not None

    def test_max_samples_limit(self):
        visualiser = CaptumImageVisualiser(method="heat_map")
        large_batch = torch.randn(64, 3, 32, 32)

        fig = visualiser.visualise(large_batch, max_samples=4)
        assert len(fig.axes) >= 4

    def test_overlay_with_inputs(self, sample_images):
        visualiser = CaptumImageVisualiser()
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(attributions, inputs=sample_images)
        assert fig is not None

    def test_save(self, sample_images, tmp_path):
        visualiser = CaptumImageVisualiser(method="heat_map")
        attributions = torch.randn_like(sample_images)
        output_path = tmp_path / "test_output.png"

        visualiser.save(attributions, output_path)
        assert output_path.exists()


class TestTabularBarChartVisualiser:
    """Test tabular visualiser"""

    def test_initialization(self):
        """Test visualiser can be initialized"""
        visualiser = TabularBarChartVisualiser()
        assert visualiser is not None

    def test_initialization_with_feature_names(self, feature_names):
        """Test initialization with feature names"""
        visualiser = TabularBarChartVisualiser(feature_names=feature_names)
        assert visualiser.feature_names == feature_names

    def test_visualize_tensor(self, sample_tabular):
        """Test visualization with torch.Tensor"""
        visualiser = TabularBarChartVisualiser()

        fig = visualiser.visualise(sample_tabular)
        assert fig is not None

    def test_visualize_numpy(self, sample_tabular):
        """Test visualization with numpy array"""
        visualiser = TabularBarChartVisualiser()
        attributions = sample_tabular.numpy()

        fig = visualiser.visualise(attributions)
        assert fig is not None

    def test_feature_names_display(self, sample_tabular, feature_names):
        """Test feature names are displayed"""
        visualiser = TabularBarChartVisualiser(feature_names=feature_names)

        fig = visualiser.visualise(sample_tabular)
        assert fig is not None

    def test_save(self, sample_tabular, tmp_path):
        """Test save functionality"""
        visualiser = TabularBarChartVisualiser()
        output_path = tmp_path / "test_tabular.png"

        visualiser.save(sample_tabular, output_path)
        assert output_path.exists()


class TestCaptumTimeSeriesVisualiser:
    """Test CaptumTimeSeriesVisualiser"""

    def test_initialization(self):
        visualiser = CaptumTimeSeriesVisualiser()
        assert visualiser is not None

    def test_visualise_requires_inputs(self, needs_captum, sample_timeseries):
        """visualise() requires inputs alongside attributions."""
        visualiser = CaptumTimeSeriesVisualiser()
        attributions = torch.randn_like(sample_timeseries)
        with pytest.raises(ValueError, match="requires `inputs`"):
            visualiser.visualise(attributions)

    def test_visualise_with_inputs(self, needs_captum, sample_timeseries):
        """Returns a Figure when inputs are supplied."""
        visualiser = CaptumTimeSeriesVisualiser()
        attributions = torch.randn_like(sample_timeseries)

        fig = visualiser.visualise(attributions, inputs=sample_timeseries)
        assert fig is not None

    def test_save(self, needs_captum, sample_timeseries, tmp_path):
        visualiser = CaptumTimeSeriesVisualiser()
        attributions = torch.randn_like(sample_timeseries)
        output_path = tmp_path / "timeseries.png"

        visualiser.save(attributions, output_path, inputs=sample_timeseries)
        assert output_path.exists()


class TestCaptumTextVisualiser:
    """Test CaptumTextVisualiser (pure matplotlib — no captum dependency)."""

    def test_initialization(self):
        visualiser = CaptumTextVisualiser()
        assert visualiser is not None

    def test_visualise_1d_tensor(self, sample_text_attributions):
        """1-D attribution tensor produces a figure."""
        visualiser = CaptumTextVisualiser()
        fig = visualiser.visualise(sample_text_attributions)
        assert fig is not None

    def test_visualise_with_token_labels(self, sample_text_attributions, token_labels):
        """Figure is produced with token labels supplied."""
        visualiser = CaptumTextVisualiser()
        fig = visualiser.visualise(sample_text_attributions, token_labels=token_labels)
        assert fig is not None

    def test_save(self, sample_text_attributions, tmp_path):
        visualiser = CaptumTextVisualiser()
        output_path = tmp_path / "text.png"
        visualiser.save(sample_text_attributions, output_path)
        assert output_path.exists()


class TestShapBarVisualiser:
    """Test ShapBarVisualiser"""

    def test_initialization(self):
        visualiser = ShapBarVisualiser()
        assert visualiser is not None

    def test_visualise_tensor(self, needs_shap, sample_tabular):
        visualiser = ShapBarVisualiser()
        fig = visualiser.visualise(sample_tabular)
        assert fig is not None

    def test_visualise_with_feature_names(self, needs_shap, sample_tabular, feature_names):
        visualiser = ShapBarVisualiser(feature_names=feature_names)
        fig = visualiser.visualise(sample_tabular)
        assert fig is not None

    def test_save(self, needs_shap, sample_tabular, tmp_path):
        visualiser = ShapBarVisualiser()
        output_path = tmp_path / "shap_bar.png"
        visualiser.save(sample_tabular, output_path)
        assert output_path.exists()


class TestShapBeeswarmVisualiser:
    """Test ShapBeeswarmVisualiser"""

    def test_initialization(self):
        visualiser = ShapBeeswarmVisualiser()
        assert visualiser is not None

    def test_visualise_tensor(self, needs_shap, sample_tabular):
        visualiser = ShapBeeswarmVisualiser()
        fig = visualiser.visualise(sample_tabular)
        assert fig is not None

    def test_save(self, needs_shap, sample_tabular, tmp_path):
        visualiser = ShapBeeswarmVisualiser()
        output_path = tmp_path / "shap_beeswarm.png"
        visualiser.save(sample_tabular, output_path)
        assert output_path.exists()


class TestShapWaterfallVisualiser:
    """Test ShapWaterfallVisualiser"""

    def test_initialization(self):
        visualiser = ShapWaterfallVisualiser()
        assert visualiser is not None

    def test_visualise_single_sample(self, needs_shap, sample_tabular):
        """Waterfall chart for sample_index=0 of the batch."""
        visualiser = ShapWaterfallVisualiser()
        fig = visualiser.visualise(sample_tabular)
        assert fig is not None

    def test_visualise_with_feature_names(self, needs_shap, sample_tabular, feature_names):
        visualiser = ShapWaterfallVisualiser(feature_names=feature_names)
        fig = visualiser.visualise(sample_tabular)
        assert fig is not None

    def test_save(self, needs_shap, sample_tabular, tmp_path):
        visualiser = ShapWaterfallVisualiser()
        output_path = tmp_path / "shap_waterfall.png"
        visualiser.save(sample_tabular, output_path)
        assert output_path.exists()


class TestShapForceVisualiser:
    """Test ShapForceVisualiser"""

    def test_initialization(self):
        visualiser = ShapForceVisualiser()
        assert visualiser is not None

    def test_visualise_single_sample(self, needs_shap, sample_tabular):
        """Force plot for sample_index=0 of the batch."""
        visualiser = ShapForceVisualiser()
        fig = visualiser.visualise(sample_tabular)
        assert fig is not None

    def test_save(self, needs_shap, sample_tabular, tmp_path):
        visualiser = ShapForceVisualiser()
        output_path = tmp_path / "shap_force.png"
        visualiser.save(sample_tabular, output_path)
        assert output_path.exists()


class TestShapImageVisualiser:
    """Test ShapImageVisualiser (restricted to GradientExplainer / DeepExplainer)."""

    def test_initialization(self):
        visualiser = ShapImageVisualiser()
        assert visualiser is not None

    def test_compatible_algorithms_restriction(self):
        assert ShapImageVisualiser.compatible_algorithms == frozenset(
            {"GradientExplainer", "DeepExplainer"}
        )

    def test_visualise_image_batch(self, needs_shap, sample_images):
        """Accepts a (B, C, H, W) attribution tensor."""
        visualiser = ShapImageVisualiser()
        attributions = torch.randn_like(sample_images)
        fig = visualiser.visualise(attributions, inputs=sample_images)
        assert fig is not None

    def test_save(self, needs_shap, sample_images, tmp_path):
        visualiser = ShapImageVisualiser()
        attributions = torch.randn_like(sample_images)
        output_path = tmp_path / "shap_image.png"
        visualiser.save(attributions, output_path, inputs=sample_images)
        assert output_path.exists()
