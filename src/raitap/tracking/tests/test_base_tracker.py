"""Unit tests for BaseTracker factory and context manager."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig

from raitap.tracking import BaseTracker


def _make_config(tracker_target: str = "MLFlowTracker") -> AppConfig:
    return cast(
        "AppConfig",
        SimpleNamespace(
            tracking=SimpleNamespace(
                _target_=tracker_target,
                output_forwarding_url="http://127.0.0.1:5000",
                log_model=False,
                open_when_done=False,
            ),
            experiment_name="test_experiment",
            fallback_output_dir=".",
        ),
    )


class TestCreateTracker:
    def test_create_tracker_instantiates_from_config(self) -> None:
        config = _make_config("raitap.tracking.MLFlowTracker")

        with patch("raitap.tracking.base_tracker.instantiate") as mock_instantiate:
            mock_class = MagicMock()
            mock_instance = MagicMock(spec=BaseTracker)
            mock_class.return_value = mock_instance
            mock_instantiate.return_value = mock_class

            tracker = BaseTracker.create_tracker(config)

            assert tracker is mock_instance
            mock_class.assert_called_once_with(config)

    def test_create_tracker_resolves_short_target_names(self) -> None:
        config = _make_config("MLFlowTracker")

        with patch("raitap.tracking.base_tracker.instantiate") as mock_instantiate:
            mock_class = MagicMock()
            mock_instance = MagicMock(spec=BaseTracker)
            mock_class.return_value = mock_instance
            mock_instantiate.return_value = mock_class

            _ = BaseTracker.create_tracker(config)

            call_args = mock_instantiate.call_args[0][0]
            assert "raitap.tracking." in call_args["_target_"]

    def test_create_tracker_raises_on_invalid_target(self) -> None:
        config = _make_config("NonExistentTracker")

        with patch("raitap.tracking.base_tracker.instantiate") as mock_instantiate:
            mock_instantiate.side_effect = Exception("Cannot instantiate")

            with pytest.raises(ValueError, match="Could not instantiate tracker"):
                _ = BaseTracker.create_tracker(config)


class MockTracker(BaseTracker):
    """Concrete implementation for testing context manager."""

    def __init__(self):
        self.terminated = False
        self.terminate_success = None

    def log_config(self) -> None:
        pass

    def log_model(self, model: Any) -> None:
        pass

    def log_dataset(self, description: dict[str, Any]) -> None:
        pass

    def log_artifacts(
        self, source_directory: str | Path | None, target_subdirectory: str | None = None
    ) -> None:
        pass

    def log_metrics(
        self, metrics: dict[str, float | int | bool], prefix: str = "performance"
    ) -> None:
        pass

    def terminate(self, successfully: bool = True) -> None:
        self.terminated = True
        self.terminate_success = successfully


class TestContextManager:
    def test_context_manager_returns_self(self) -> None:
        tracker = MockTracker()

        with tracker as ctx:
            assert ctx is tracker

    def test_context_manager_calls_terminate_on_exit(self) -> None:
        tracker = MockTracker()

        with tracker:
            assert not tracker.terminated

        assert tracker.terminated
        assert tracker.terminate_success is True

    def test_context_manager_calls_terminate_with_false_on_exception(self) -> None:
        tracker = MockTracker()

        with pytest.raises(ValueError), tracker:
            raise ValueError("Test error")

        assert tracker.terminated
        assert tracker.terminate_success is False

    def test_context_manager_does_not_suppress_exceptions(self) -> None:
        tracker = MockTracker()

        with pytest.raises(ValueError, match="Test error"), tracker:
            raise ValueError("Test error")

    def test_context_manager_terminates_even_after_exception(self) -> None:
        tracker = MockTracker()

        try:
            with tracker:
                raise RuntimeError("Something went wrong")
        except RuntimeError:
            pass

        assert tracker.terminated
        assert tracker.terminate_success is False
