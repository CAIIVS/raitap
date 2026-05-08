"""Focused mock-driven tests for MLFlowTracker."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from raitap.models.backend import OnnxBackend, TorchBackend

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from raitap.configs.schema import AppConfig

from raitap.tracking.mlflow_tracker import (
    DEFAULT_MLFLOW_UI_HOST,
    DEFAULT_MLFLOW_UI_PORT,
    MLFlowTracker,
)


def _make_config(
    url: str | None = "http://127.0.0.1:5000",
    open_when_done: bool = False,
    log_model: bool = False,
    backend_store_uri: str | None = None,
    default_artifact_root: str | None = None,
) -> AppConfig:
    return cast(
        "AppConfig",
        SimpleNamespace(
            tracking=SimpleNamespace(
                _target_="MLFlowTracker",
                output_forwarding_url=url,
                backend_store_uri=backend_store_uri,
                default_artifact_root=default_artifact_root,
                log_model=log_model,
                open_when_done=open_when_done,
            ),
            experiment_name="test_experiment",
            _output_root=".",
        ),
    )


@pytest.fixture
def mock_mlflow() -> Generator[MagicMock]:
    mock = MagicMock()
    # Mocking submodules if needed
    mock.onnx = MagicMock()
    mock.pytorch = MagicMock()
    mock.entities = MagicMock()
    mock.entities.RunStatus = MagicMock()
    mock.entities.RunStatus.to_string = lambda x: x
    with patch.dict("sys.modules", {"mlflow": mock, "mlflow.entities": mock.entities}):
        yield mock


@pytest.fixture
def mock_subprocess() -> Generator[MagicMock]:
    with patch("subprocess.Popen") as mock:
        yield mock


@pytest.fixture
def tracker(mock_mlflow: MagicMock) -> MLFlowTracker:
    # Avoid server startup by setting tracking_uri to something non-http
    config = _make_config(url="./mlruns")
    with patch("raitap.tracking.mlflow_tracker.MLFlowTracker._ensure_server_running"):
        return MLFlowTracker(config)


@pytest.fixture
def saved_onnx_model(tmp_path: Path) -> Path:
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    path = tmp_path / "tracking-model.onnx"
    weight = torch.full((2, 2), 0.5, dtype=torch.float32).numpy()
    bias = torch.tensor([0.1, -0.1], dtype=torch.float32).numpy()
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["input", "weight", "bias"], ["output"])],
        "tracking_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 2])],
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


def test_mlflow_tracker_init_starts_run(mock_mlflow: MagicMock) -> None:
    config = _make_config(url="http://127.0.0.1:5000")
    with patch("raitap.tracking.mlflow_tracker.MLFlowTracker._ensure_server_running"):
        tracker = MLFlowTracker(config)

    mock_mlflow.set_tracking_uri.assert_called_once_with("http://127.0.0.1:5000")
    mock_mlflow.set_experiment.assert_called_once_with("test_experiment")
    mock_mlflow.start_run.assert_called_once_with(run_name="test_experiment")
    assert tracker.tracking_uri == "http://127.0.0.1:5000"


def test_mlflow_tracker_defaults_to_sqlite_backend(mock_mlflow: MagicMock) -> None:
    config = _make_config(url=None)
    with patch("raitap.tracking.mlflow_tracker.MLFlowTracker._ensure_server_running"):
        tracker = MLFlowTracker(config)

    mock_mlflow.set_tracking_uri.assert_called_once_with("sqlite:///mlflow/mlflow.db")
    assert tracker.tracking_uri == "sqlite:///mlflow/mlflow.db"
    assert tracker.backend_store_uri == "sqlite:///mlflow/mlflow.db"
    assert tracker.default_artifact_root == "./mlflow/artifacts"


def test_mlflow_tracker_creates_direct_sqlite_experiment_with_artifact_root(
    mock_mlflow: MagicMock,
) -> None:
    mock_mlflow.get_experiment_by_name.return_value = None
    config = _make_config(url="sqlite:///mlflow/mlflow.db")

    with patch("raitap.tracking.mlflow_tracker.MLFlowTracker._ensure_server_running"):
        MLFlowTracker(config)

    mock_mlflow.set_tracking_uri.assert_called_once_with("sqlite:///mlflow/mlflow.db")
    mock_mlflow.create_experiment.assert_called_once_with(
        "test_experiment",
        artifact_location="./mlflow/artifacts",
    )
    mock_mlflow.set_experiment.assert_called_once_with("test_experiment")


def test_mlflow_tracker_starts_local_server_with_sqlite_backend(
    mock_mlflow: MagicMock,
    mock_subprocess: MagicMock,
) -> None:
    config = _make_config(
        url="http://127.0.0.1:5000",
        backend_store_uri="sqlite:///mlflow/mlflow.db",
        default_artifact_root="./mlflow/artifacts",
    )

    with (
        patch("raitap.tracking.mlflow_tracker.MLFlowTracker._is_port_open", return_value=False),
        patch(
            "raitap.tracking.mlflow_tracker.MLFlowTracker._wait_for_port_ready",
            return_value=True,
        ),
    ):
        MLFlowTracker(config)

    mock_subprocess.assert_called_once_with(
        [
            "mlflow",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            "5000",
            "--backend-store-uri",
            "sqlite:///mlflow/mlflow.db",
            "--default-artifact-root",
            "./mlflow/artifacts",
        ],
        stdout=-3,
        stderr=-3,
        start_new_session=True,
    )


def test_mlflow_tracker_opens_sqlite_ui(
    mock_mlflow: MagicMock,
    mock_subprocess: MagicMock,
) -> None:
    config = _make_config(url="sqlite:///mlflow/mlflow.db")
    with patch("raitap.tracking.mlflow_tracker.MLFlowTracker._ensure_server_running"):
        tracker = MLFlowTracker(config)

    with (
        patch("raitap.tracking.mlflow_tracker.MLFlowTracker._is_port_open", return_value=False),
        patch(
            "raitap.tracking.mlflow_tracker.MLFlowTracker._wait_for_port_ready",
            return_value=True,
        ),
        patch("webbrowser.open") as mock_open,
    ):
        tracker._open_mlflow_ui()

    ui_url = f"http://{DEFAULT_MLFLOW_UI_HOST}:{DEFAULT_MLFLOW_UI_PORT}"
    mock_subprocess.assert_called_once_with(
        [
            "mlflow",
            "ui",
            "--backend-store-uri",
            "sqlite:///mlflow/mlflow.db",
            "--port",
            str(DEFAULT_MLFLOW_UI_PORT),
        ],
        stdout=-3,
        stderr=-3,
        start_new_session=True,
    )
    mock_open.assert_called_once_with(ui_url)


def test_mlflow_tracker_warns_when_reusing_existing_sqlite_ui_port(
    mock_mlflow: MagicMock,
    mock_subprocess: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _make_config(url="sqlite:///mlflow/mlflow.db")
    with patch("raitap.tracking.mlflow_tracker.MLFlowTracker._ensure_server_running"):
        tracker = MLFlowTracker(config)

    with (
        patch("raitap.tracking.mlflow_tracker.MLFlowTracker._is_port_open", return_value=True),
        patch("webbrowser.open") as mock_open,
        caplog.at_level("WARNING", logger="raitap.tracking.mlflow_tracker"),
    ):
        tracker._open_mlflow_ui()

    mock_subprocess.assert_not_called()
    mock_open.assert_called_once_with("http://127.0.0.1:5000")
    assert "Reusing existing MLflow UI at http://127.0.0.1:5000" in caplog.text
    assert "sqlite:///mlflow/mlflow.db" in caplog.text
    assert "may use a different backend store" in caplog.text


def test_mlflow_tracker_terminate_keeps_opened_ui_running(
    mock_mlflow: MagicMock,
    mock_subprocess: MagicMock,
) -> None:
    config = _make_config(url="sqlite:///mlflow/mlflow.db", open_when_done=True)
    with patch("raitap.tracking.mlflow_tracker.MLFlowTracker._ensure_server_running"):
        tracker = MLFlowTracker(config)

    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_subprocess.return_value = mock_process

    with (
        patch("raitap.tracking.mlflow_tracker.MLFlowTracker._is_port_open", return_value=False),
        patch(
            "raitap.tracking.mlflow_tracker.MLFlowTracker._wait_for_port_ready",
            return_value=True,
        ),
        patch("webbrowser.open"),
    ):
        tracker.terminate(successfully=True)

    mock_mlflow.end_run.assert_called_once()
    mock_process.terminate.assert_not_called()
    mock_process.kill.assert_not_called()


def test_mlflow_tracker_terminate_cleans_up_when_not_opening_ui(
    mock_mlflow: MagicMock,
) -> None:
    config = _make_config(url="sqlite:///mlflow/mlflow.db", open_when_done=False)
    with patch("raitap.tracking.mlflow_tracker.MLFlowTracker._ensure_server_running"):
        tracker = MLFlowTracker(config)

    mock_process = MagicMock()
    mock_process.poll.return_value = None
    tracker._ui_process = mock_process

    tracker.terminate(successfully=True)

    mock_mlflow.end_run.assert_called_once()
    mock_process.terminate.assert_called_once()


def test_mlflow_tracker_terminate_success(tracker: MLFlowTracker, mock_mlflow: MagicMock) -> None:
    tracker.terminate(successfully=True)
    # FINISHED is the default for success
    mock_mlflow.end_run.assert_called_once()
    # Check that status was called with something (status string depends on mlflow.entities mock)


def test_mlflow_tracker_terminate_failure(tracker: MLFlowTracker, mock_mlflow: MagicMock) -> None:
    tracker.terminate(successfully=False)
    mock_mlflow.end_run.assert_called_once()


def test_mlflow_tracker_log_config(tracker: MLFlowTracker, mock_mlflow: MagicMock) -> None:
    tracker.log_config()
    mock_mlflow.log_dict.assert_called()
    mock_mlflow.log_params.assert_called()


def test_mlflow_tracker_cleanup_subprocesses(tracker: MLFlowTracker) -> None:
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # Still running
    tracker._server_process = mock_proc

    tracker._cleanup_subprocesses()

    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once()


def test_mlflow_tracker_log_metrics(tracker: MLFlowTracker, mock_mlflow: MagicMock) -> None:
    metrics = {"acc": 0.95, "loss": 0.1}
    tracker.log_metrics(metrics, prefix="eval")

    mock_mlflow.log_metrics.assert_called_once_with({"eval.acc": 0.95, "eval.loss": 0.1})


def test_mlflow_tracker_log_model_uses_pytorch_flavor(
    tracker: MLFlowTracker, mock_mlflow: MagicMock
) -> None:
    backend = TorchBackend(torch.nn.Linear(2, 1))

    tracker.log_model(backend)

    mock_mlflow.pytorch.log_model.assert_called_once()
    assert mock_mlflow.onnx.log_model.call_count == 0


def test_mlflow_tracker_log_model_uses_onnx_flavor(
    tracker: MLFlowTracker,
    mock_mlflow: MagicMock,
    saved_onnx_model: Path,
) -> None:
    backend = OnnxBackend.from_path(saved_onnx_model)

    tracker.log_model(backend)

    mock_mlflow.onnx.log_model.assert_called_once()
    assert mock_mlflow.pytorch.log_model.call_count == 0
