"""Unit tests for the Model class."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig

from raitap.models import Model


def _make_config(source: str) -> AppConfig:
    return cast(
        "AppConfig",
        SimpleNamespace(
            model=SimpleNamespace(
                source=source,
            )
        ),
    )


class TestModelConstructor:
    def test_model_loads_torchvision_model_by_name(self) -> None:
        config = _make_config("resnet18")

        model = Model(config)

        assert isinstance(model.network, torch.nn.Module)
        assert model.network.__class__.__name__ == "ResNet"

    def test_model_accepts_local_pth_file_path(self, tmp_path: Path) -> None:
        dummy_model = torch.nn.Linear(10, 5)
        model_path = tmp_path / "model.pth"
        torch.save(dummy_model.state_dict(), model_path)

        config = _make_config(str(model_path))

        with patch("raitap.models.model.CONVERTERS") as mock_converters:
            mock_converter = MagicMock()
            mock_converter.convert.return_value = dummy_model
            mock_converters.get.return_value = mock_converter

            model = Model(config)

            mock_converter.convert.assert_called_once()
            assert isinstance(model.network, torch.nn.Module)

    def test_model_raises_if_source_is_none(self) -> None:
        config = _make_config("")

        with pytest.raises(ValueError, match="No model specified"):
            Model(config)

    def test_model_raises_if_source_does_not_exist_and_not_known(self) -> None:
        config = _make_config("unknown_model_name")

        with pytest.raises(ValueError, match="neither an existing path nor a known"):
            Model(config)

    def test_model_raises_if_file_not_found(self) -> None:
        config = _make_config("/nonexistent/path/model.pth")

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            Model(config)


class TestModelLog:
    def test_log_calls_tracker_log_model(self) -> None:
        config = _make_config("resnet18")
        model = Model(config)

        tracker = MagicMock()
        model.log(tracker)

        tracker.log_model.assert_called_once_with(model.network)

    def test_log_passes_network_to_tracker(self) -> None:
        config = _make_config("resnet18")
        model = Model(config)

        tracker = MagicMock()
        model.log(tracker)

        logged_model = tracker.log_model.call_args[0][0]
        assert logged_model is model.network
        assert isinstance(logged_model, torch.nn.Module)
