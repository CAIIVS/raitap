"""Reusable model and config builders for tests.

Replaces the per-file ``_TinyClassifier`` / ``_FakeXpuModule`` / ``_make_config``
duplicates. Keep these intentionally tiny and CPU-only.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn


def make_tiny_classifier(*, in_channels: int = 3, num_classes: int = 2, seed: int = 0) -> nn.Module:
    """A deterministic conv classifier for attribution/parity tests."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Conv2d(in_channels, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, num_classes),
    )
    model.eval()
    return model


def make_tiny_mlp(*, in_features: int = 10, num_classes: int = 2, seed: int = 0) -> nn.Module:
    """A deterministic MLP for tabular attribution/parity tests."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(in_features, 16), nn.ReLU(), nn.Linear(16, num_classes))
    model.eval()
    return model


def make_app_config(**overrides: Any) -> SimpleNamespace:
    """Minimal AppConfig stand-in. Pass keyword overrides for any attribute."""
    base: dict[str, Any] = {"experiment_name": "test", "hardware": "cpu", "_output_root": "."}
    base.update(overrides)
    return SimpleNamespace(**base)
