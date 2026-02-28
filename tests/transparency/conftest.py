"""Shared fixtures for transparency module tests"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


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


@pytest.fixture
def sample_images():
    """Sample image batch"""
    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def sample_tabular():
    """Sample tabular data"""
    return torch.randn(8, 10)


@pytest.fixture
def feature_names():
    """Feature names for tabular data"""
    return [f"feature_{i}" for i in range(10)]
