"""Tests for visualiser implementations"""

from __future__ import annotations

import numpy as np
import torch

from raitap.transparency.visualisers import ImageHeatmapvisualiser, TabularBarChartvisualiser


class TestImageHeatmapvisualiser:
    """Test image visualiser"""

    def test_initialization(self):
        """Test visualiser can be initialized"""
        visualiser = ImageHeatmapvisualiser()
        assert visualiser is not None

    def test_visualize_tensor(self, sample_images):
        """Test visualization with torch.Tensor"""
        visualiser = ImageHeatmapvisualiser()
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(attributions)
        assert fig is not None
        assert len(fig.axes) == 8  # 4 samples (each with main axes + colorbar axes)

    def test_visualize_numpy(self, sample_images):
        """Test visualization with numpy array"""
        visualiser = ImageHeatmapvisualiser()
        attributions = np.random.randn(*sample_images.shape)

        fig = visualiser.visualise(attributions)
        assert fig is not None

    def test_max_samples_limit(self):
        """Test batch size limiting"""
        visualiser = ImageHeatmapvisualiser()
        large_batch = torch.randn(64, 3, 32, 32)

        fig = visualiser.visualise(large_batch, max_samples=8)
        assert len(fig.axes) == 16  # 8 images (each with main axes + colorbar axes)

    def test_channel_aggregation(self):
        """Test aggregation across channels"""
        visualiser = ImageHeatmapvisualiser()
        attributions = torch.randn(2, 3, 32, 32)  # (B, C, H, W)

        fig = visualiser.visualise(attributions)
        assert fig is not None
        assert len(fig.axes) == 4  # 2 images (each with main axes + colorbar axes)

    def test_save(self, sample_images, tmp_path):
        """Test save functionality"""
        visualiser = ImageHeatmapvisualiser()
        attributions = torch.randn_like(sample_images)
        output_path = tmp_path / "test_output.png"

        visualiser.save(attributions, output_path)
        assert output_path.exists()

    def test_overlay_with_inputs(self, sample_images):
        """Test overlay with original images"""
        visualiser = ImageHeatmapvisualiser()
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(attributions, inputs=sample_images)
        assert fig is not None


class TestTabularBarChartvisualiser:
    """Test tabular visualiser"""

    def test_initialization(self):
        """Test visualiser can be initialized"""
        visualiser = TabularBarChartvisualiser()
        assert visualiser is not None

    def test_initialization_with_feature_names(self, feature_names):
        """Test initialization with feature names"""
        visualiser = TabularBarChartvisualiser(feature_names=feature_names)
        assert visualiser.feature_names == feature_names

    def test_visualize_tensor(self, sample_tabular):
        """Test visualization with torch.Tensor"""
        visualiser = TabularBarChartvisualiser()

        fig = visualiser.visualise(sample_tabular)
        assert fig is not None

    def test_visualize_numpy(self, sample_tabular):
        """Test visualization with numpy array"""
        visualiser = TabularBarChartvisualiser()
        attributions = sample_tabular.numpy()

        fig = visualiser.visualise(attributions)
        assert fig is not None

    def test_feature_names_display(self, sample_tabular, feature_names):
        """Test feature names are displayed"""
        visualiser = TabularBarChartvisualiser(feature_names=feature_names)

        fig = visualiser.visualise(sample_tabular)
        assert fig is not None

    def test_save(self, sample_tabular, tmp_path):
        """Test save functionality"""
        visualiser = TabularBarChartvisualiser()
        output_path = tmp_path / "test_tabular.png"

        visualiser.save(sample_tabular, output_path)
        assert output_path.exists()
