"""Shared fixtures for transparency module tests"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
import torch.nn as nn

from raitap.models.backend import OnnxBackend

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Optional-dependency skip fixtures
# Usage: reference via ``@pytest.mark.usefixtures("needs_captum")`` (or
# ``needs_shap``) on tests that run code importing that package. Skips when the
# extra is not installed (`uv sync --extra captum` / `--extra shap`).
# ---------------------------------------------------------------------------


@pytest.fixture
def needs_captum() -> None:
    """Skip the test if captum is not installed."""
    pytest.importorskip("captum")


@pytest.fixture
def needs_shap() -> None:
    """Skip the test if shap is not installed."""
    pytest.importorskip("shap")


@pytest.fixture
def needs_alibi() -> None:
    """Skip the test if alibi is not installed."""
    pytest.importorskip("alibi")


@pytest.fixture
def needs_onnx() -> None:
    """Skip the test if ONNX dependencies are not installed."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")


@pytest.fixture(autouse=True)
def isolate_transparency_test_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep fallback transparency artifacts inside pytest's temporary directory."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def reset_alibi_bsl_warning_flag() -> None:
    """So license warning tests see a fresh one-time flag each test."""
    from raitap.transparency import factory as transparency_factory

    transparency_factory._ALIBI_BSL_WARNING_EMITTED = False


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_cnn() -> nn.Module:
    """Simple CNN for testing"""
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10)
    )
    model.eval()
    return model


@pytest.fixture
def simple_mlp() -> nn.Module:
    """Simple MLP for tabular data testing"""
    model = nn.Sequential(
        nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2)
    )
    model.eval()
    return model


class SimpleTimeSeriesClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(8, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        channels_first = inputs.transpose(1, 2)
        features = self.conv(channels_first)
        features = self.relu(features)
        pooled = self.pool(features).squeeze(-1)
        return self.head(pooled)


@pytest.fixture
def simple_timeseries_model() -> nn.Module:
    """Simple time-series model that consumes inputs shaped (batch, time, channels)."""
    model = SimpleTimeSeriesClassifier()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_images() -> torch.Tensor:
    """Sample image batch: (batch, channels, height, width)."""
    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def sample_tabular() -> torch.Tensor:
    """Sample tabular data: (batch, features)."""
    return torch.randn(8, 10)


@pytest.fixture
def sample_timeseries() -> torch.Tensor:
    """Sample time-series batch: (batch, time_steps, channels)."""
    return torch.randn(4, 50, 3)


@pytest.fixture
def sample_text_attributions() -> torch.Tensor:
    """1-D per-token attribution scores."""
    return torch.randn(15)


@pytest.fixture
def feature_names() -> list[str]:
    """Feature names for tabular data"""
    return [f"feature_{i}" for i in range(10)]


@pytest.fixture
def token_labels() -> list[str]:
    """Token labels for text attribution tests."""
    return [f"tok_{i}" for i in range(15)]


@pytest.fixture
def onnx_linear_path(tmp_path: Path, needs_onnx: None) -> Path:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    path = tmp_path / "linear.onnx"

    weight = np.full((10, 2), 0.1, dtype=np.float32)
    bias = np.array([0.05, -0.05], dtype=np.float32)

    graph = helper.make_graph(
        [
            helper.make_node("Gemm", ["input", "weight", "bias"], ["output"]),
        ],
        "linear_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 10])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 2])],
        [
            numpy_helper.from_array(weight, name="weight"),
            numpy_helper.from_array(bias, name="bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


@pytest.fixture
def onnx_linear_backend(onnx_linear_path: Path) -> OnnxBackend:
    return OnnxBackend.from_path(onnx_linear_path)
