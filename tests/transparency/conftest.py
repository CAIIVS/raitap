"""Shared fixtures for transparency module tests"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Optional-dependency skip fixtures
# Usage: add `needs_captum` or `needs_shap` as a parameter to any test that
# requires the respective library.  The test is automatically skipped when the
# library is not installed instead of raising an ImportError.
# ---------------------------------------------------------------------------


@pytest.fixture
def needs_captum():
    """Skip the test if captum is not installed."""
    pytest.importorskip("captum")


@pytest.fixture
def needs_shap():
    """Skip the test if shap is not installed."""
    pytest.importorskip("shap")


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_cnn():
    """Simple CNN for testing"""
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10)
    )
    model.eval()
    return model


@pytest.fixture
def simple_mlp():
    """Simple MLP for tabular data testing"""
    model = nn.Sequential(
        nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2)
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_images():
    """Sample image batch: (batch, channels, height, width)."""
    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def sample_tabular():
    """Sample tabular data: (batch, features)."""
    return torch.randn(8, 10)


@pytest.fixture
def sample_timeseries():
    """Sample time-series batch: (batch, time_steps, channels)."""
    return torch.randn(4, 50, 3)


@pytest.fixture
def sample_text_attributions():
    """1-D per-token attribution scores."""
    return torch.randn(15)


@pytest.fixture
def feature_names():
    """Feature names for tabular data"""
    return [f"feature_{i}" for i in range(10)]


@pytest.fixture
def token_labels():
    """Token labels for text attribution tests."""
    return [f"tok_{i}" for i in range(15)]
