"""Reusable model and config builders for tests.

Replaces the per-file ``_TinyClassifier`` / ``_FakeXpuModule`` / ``_make_config``
duplicates. Keep these intentionally tiny and CPU-only.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from raitap.configs.schema import AppConfig
from raitap.models.base_backend import ModelBackend
from raitap.types import Capability, TaskKind


def make_tiny_classifier(
    *, in_channels: int = 3, num_classes: int = 2, seed: int = 0
) -> nn.Sequential:
    """A deterministic conv classifier for attribution/parity tests.

    Returns ``nn.Sequential`` (not just ``nn.Module``) so callers can index
    layers, e.g. ``model[0]`` for a GradCAM target layer.
    """
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


def make_tiny_mlp(*, in_features: int = 10, num_classes: int = 2, seed: int = 0) -> nn.Sequential:
    """A deterministic MLP for tabular attribution/parity tests."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(in_features, 16), nn.ReLU(), nn.Linear(16, num_classes))
    model.eval()
    return model


def make_pixel_linear_classifier(
    *, channels: int = 3, hw: int = 8, num_classes: int = 3, seed: int = 0
) -> nn.Module:
    """A single ``Linear`` over flattened CHW pixels.

    Small enough that even iterative adversarial attacks finish in well under a
    second on CPU. Used by the robustness assessor/e2e tests, which need a
    fixed-input-size differentiable classifier (not the size-agnostic conv of
    ``make_tiny_classifier``).
    """
    torch.manual_seed(seed)

    class _PixelLinearClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = nn.Linear(channels * hw * hw, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layer(x.flatten(1))

    model = _PixelLinearClassifier()
    model.eval()
    return model


def make_app_config(**overrides: Any) -> DictConfig:
    """A faithful ``AppConfig`` stand-in for tests.

    Returns a struct-mode OmegaConf config typed by ``AppConfig`` — the same
    shape Hydra hands prod at runtime. Attribute access resolves declared
    fields (with schema defaults) and raises on undeclared ones, so tests
    model valid configs and prod can read fields directly without defensive
    ``getattr``. Pass flat or nested keyword overrides
    (``make_app_config(model={"source": "resnet18"})``).
    """
    merged = OmegaConf.merge(OmegaConf.structured(AppConfig), overrides)
    return cast("DictConfig", merged)


def make_fake_backend(
    *,
    provides: frozenset[Capability] = frozenset(),
    task_kind: TaskKind = TaskKind.classification,
    hardware_label: str = "cpu",
) -> ModelBackend:
    """A minimal concrete ``ModelBackend`` for tests that feed a backend into
    prod read paths (``backend.provides`` / ``.task_kind`` / ``.device``).

    Replaces ``SimpleNamespace`` backend fakes so those reads resolve against
    the real ABC contract instead of defensive ``getattr``.
    """
    # Aliased to avoid the class-body self-reference trap: `provides = provides`
    # inside the class body would make `provides` a class-local name for the
    # whole statement, so the RHS lookup raises NameError instead of closing
    # over the function argument.
    _provides = provides

    class _FakeBackend(ModelBackend):
        provides = _provides  # type: ignore[misc]

        @property
        def hardware_label(self) -> str:
            return hardware_label

        @property
        def task_kind(self) -> TaskKind:
            return task_kind

        def __call__(self, inputs: Any, **kwargs: Any) -> Any:
            return inputs

    return _FakeBackend()
