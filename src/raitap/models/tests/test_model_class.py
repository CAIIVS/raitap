"""Unit tests for the Model class."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig

from raitap.models import Model
from raitap.models.backend import TorchBackend


def _make_config(
    source: str,
    *,
    arch: str | None = None,
    num_classes: int | None = None,
    pretrained: bool = False,
    hardware: str = "cpu",
) -> AppConfig:
    return cast(
        "AppConfig",
        SimpleNamespace(
            model=SimpleNamespace(
                source=source,
                arch=arch,
                num_classes=num_classes,
                pretrained=pretrained,
            ),
            hardware=hardware,
        ),
    )


class TestModelConstructor:
    def test_model_loads_torchvision_model_by_name(self) -> None:
        config = _make_config("resnet18")

        model = Model(config)

        assert isinstance(model.backend, TorchBackend)
        assert model.backend.as_model_for_explanation().__class__.__name__ == "ResNet"

    def test_model_loads_from_local_pth_file(self, tmp_path: Path) -> None:
        dummy_model = torch.nn.Linear(10, 5)
        model_path = tmp_path / "model.pth"
        torch.save(dummy_model, model_path)

        config = _make_config(str(model_path))
        with pytest.warns(DeprecationWarning, match="pickled nn.Module"):
            model = Model(config)

        assert isinstance(model.backend.as_model_for_explanation(), torch.nn.Module)

    def test_model_loads_from_state_dict(self, tmp_path: Path) -> None:
        from torchvision import models as tv_models

        ref = tv_models.resnet18(weights=None, num_classes=2)
        model_path = tmp_path / "weights.pth"
        torch.save(ref.state_dict(), model_path)

        config = _make_config(str(model_path), arch="resnet18", num_classes=2)
        model = Model(config)

        loaded = model.backend.as_model_for_explanation()
        assert loaded.__class__.__name__ == "ResNet"
        # State dict round-trips: parameters should compare equal.
        for (n1, p1), (n2, p2) in zip(
            ref.state_dict().items(), loaded.state_dict().items(), strict=True
        ):
            assert n1 == n2
            assert torch.equal(p1, p2.cpu())

    def test_model_state_dict_without_arch_raises(self, tmp_path: Path) -> None:
        ref = torch.nn.Linear(4, 2)
        model_path = tmp_path / "weights.pth"
        torch.save(ref.state_dict(), model_path)

        config = _make_config(str(model_path))  # no arch / num_classes
        with pytest.raises(ValueError, match="State-dict loading requires"):
            Model(config)

    def test_model_state_dict_with_unknown_arch_raises(self, tmp_path: Path) -> None:
        ref = torch.nn.Linear(4, 2)
        model_path = tmp_path / "weights.pth"
        torch.save(ref.state_dict(), model_path)

        config = _make_config(str(model_path), arch="not_a_model", num_classes=2)
        with pytest.raises(ValueError, match="not a known torchvision model"):
            Model(config)

    def test_model_loads_torchscript_file(self, tmp_path: Path) -> None:
        ref = torch.nn.Linear(4, 2).eval()
        scripted = torch.jit.script(ref)
        model_path = tmp_path / "scripted.pt"
        scripted.save(str(model_path))

        config = _make_config(str(model_path))
        model = Model(config)

        loaded = model.backend.as_model_for_explanation()
        assert isinstance(loaded, torch.jit.ScriptModule)
        # Sanity: forward pass produces same output as the eager reference.
        x = torch.randn(3, 4)
        torch.testing.assert_close(loaded(x), ref(x))

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

        tracker.log_model.assert_called_once_with(model.backend)

    def test_log_passes_backend_to_tracker(self) -> None:
        config = _make_config("resnet18")
        model = Model(config)

        tracker = MagicMock()
        model.log(tracker)

        logged_model = tracker.log_model.call_args[0][0]
        assert logged_model is model.backend
