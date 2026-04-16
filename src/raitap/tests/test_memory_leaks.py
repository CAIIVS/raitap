"""Tests that verify computation-graph and GPU memory are not retained after key operations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import torch

from raitap.transparency.results import ExplanationPayloadKind, ExplanationResult

if TYPE_CHECKING:
    from raitap.data import Data
    from raitap.models import Model

# ---------------------------------------------------------------------------
# ExplanationResult: tensors must be detached and on CPU after construction
# ---------------------------------------------------------------------------


def test_explanation_result_detaches_tensors() -> None:
    """Tensors with live grad_fn are detached so the computation graph is not pinned."""
    # squeeze(1) preserves requires_grad — use shape (N,1) to get a float prediction tensor
    attributions = torch.randn(2, 3, requires_grad=True)
    inputs = torch.randn(2, 3, requires_grad=True)

    res = ExplanationResult(
        attributions=attributions,
        inputs=inputs,
        run_dir=Path("/tmp/test"),
        experiment_name="test",
        explainer_target="test_target",
        algorithm="test_alg",
        payload_kind=ExplanationPayloadKind.ATTRIBUTIONS,
    )

    assert not res.attributions.requires_grad, "attributions must not retain grad"
    assert not res.inputs.requires_grad, "inputs must not retain grad"


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
        ),
    ],
)
def test_explanation_result_moves_tensors_to_cpu(device: str) -> None:
    """Tensors are always stored on CPU regardless of which device they came from."""
    attributions = torch.randn(2, 3).to(device)
    inputs = torch.randn(2, 3).to(device)

    res = ExplanationResult(
        attributions=attributions,
        inputs=inputs,
        run_dir=Path("/tmp/test"),
        experiment_name="test",
        explainer_target="test_target",
        algorithm="test_alg",
        payload_kind=ExplanationPayloadKind.ATTRIBUTIONS,
    )

    assert res.attributions.device.type == "cpu"
    assert res.inputs.device.type == "cpu"


# ---------------------------------------------------------------------------
# _resolve_explainer_runtime_kwargs: predicted target must be detached
# ---------------------------------------------------------------------------


def test_resolve_explainer_runtime_kwargs_detaches_target() -> None:
    """The auto_pred target passed to explainers must not carry a computation graph.

    Uses a (N, 1) forward output so metrics_prediction_pair returns squeeze(1),
    a float tensor that *would* keep requires_grad without an explicit .detach().
    """
    from raitap.run.pipeline import _resolve_explainer_runtime_kwargs

    # Single-output shape: metrics_prediction_pair returns output.squeeze(1), a float
    # tensor that inherits requires_grad from the source without .detach().
    forward_output = torch.randn(2, 1)
    forward_output.requires_grad_()

    explainer_cfg = SimpleNamespace(call={"target": "auto_pred"})

    result = _resolve_explainer_runtime_kwargs(explainer_cfg, forward_output=forward_output)

    target = result["target"]
    assert not target.requires_grad, "explainer target must be detached from the graph"
    assert target.device == forward_output.device, "target must stay on the original device"


# ---------------------------------------------------------------------------
# _run_without_tracking: RunOutputs.forward_output must be CPU and detached
# ---------------------------------------------------------------------------


def test_run_without_tracking_forward_output_is_cpu_and_detached() -> None:
    """The forward output stored in RunOutputs must always be CPU-resident and detached."""
    import torch.nn as nn

    from raitap.run.pipeline import _run_without_tracking

    net = nn.Linear(4, 2, bias=False)

    class _Backend:
        hardware_label = "cpu"

        def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
            return x

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return net(x)

    from typing import cast

    model = cast("Model", SimpleNamespace(backend=_Backend()))
    data = cast("Data", SimpleNamespace(tensor=torch.randn(2, 4), labels=None, sample_ids=None))
    # Transparency must be dict-like (supports .items() and [key] access)
    config = MagicMock()
    config.transparency = {"explainer1": SimpleNamespace()}

    fake_explanation = MagicMock()
    fake_explanation.visualise.return_value = []

    with (
        patch("raitap.run.pipeline.metrics_run_enabled", return_value=False),
        patch("raitap.run.pipeline.Explanation", return_value=fake_explanation),
    ):
        outputs = _run_without_tracking(config, model, data)

    assert not outputs.forward_output.requires_grad, "forward_output must be detached"
    assert outputs.forward_output.device.type == "cpu", "forward_output must be on CPU"


# ---------------------------------------------------------------------------
# MLFlowTracker: terminate() must clean up spawned subprocesses
# ---------------------------------------------------------------------------


def test_mlflow_tracker_terminate_cleans_up_subprocesses() -> None:
    """terminate() is responsible for subprocess cleanup; __del__ must not be relied on."""
    config = MagicMock()
    config.experiment_name = "test"

    mlflow_mock = MagicMock()
    mlflow_mock.entities = MagicMock()
    mlflow_mock.entities.RunStatus = MagicMock()
    mlflow_mock.entities.RunStatus.to_string = lambda x: x
    with (
        patch.dict("sys.modules", {"mlflow": mlflow_mock, "mlflow.entities": mlflow_mock.entities}),
        patch("raitap.tracking.mlflow_tracker.MLFlowTracker._is_port_open", return_value=False),
        patch(
            "raitap.tracking.mlflow_tracker.MLFlowTracker._wait_for_port_ready",
            return_value=True,
        ),
        patch("subprocess.Popen") as mock_popen,
        patch(
            "raitap.tracking.mlflow_tracker._tracking_dict",
            return_value={
                "output_forwarding_url": "http://127.0.0.1:5000",
                "open_when_done": False,
            },
        ),
    ):
        from raitap.tracking.mlflow_tracker import MLFlowTracker

        mock_process = MagicMock()
        mock_process.poll.return_value = None  # process is still running
        mock_popen.return_value = mock_process

        tracker = MLFlowTracker(config)
        assert tracker._server_process is not None

        tracker.terminate(successfully=True)

        mock_process.terminate.assert_called_once()
