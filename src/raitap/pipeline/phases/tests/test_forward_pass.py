"""Tests for :func:`raitap.pipeline.phases.forward_pass.forward_pass`.

Covers:
- Detection forward pass with a ragged list of differently-sized tensors.
- Classification forward pass regression (dense tensor, unchanged path).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from raitap.models.base_backend import ModelBackend
from raitap.pipeline.phases.forward_pass import forward_pass
from raitap.task_families.base import ForwardContext
from raitap.task_families.registry import resolve_task_family
from raitap.types import Capability, TaskKind

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _make_config(batch_size: int = 32) -> Any:
    """Return a minimal config stub accepted by resolve_forward_batch_size."""
    return SimpleNamespace(
        run=SimpleNamespace(forward_batch_size=batch_size),
        data=SimpleNamespace(forward_batch_size=batch_size),
    )


# ---------------------------------------------------------------------------
# Fake backends
# ---------------------------------------------------------------------------


class _FakeDetectionBackend(ModelBackend):
    """Minimal detection backend stub that records the inputs it receives."""

    provides = frozenset({Capability.AUTOGRAD})

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        self._task_kind = TaskKind.detection
        self.received_inputs: list[Any] = []

    @property
    def task_kind(self) -> TaskKind:
        return self._task_kind

    @property
    def hardware_label(self) -> str:
        return "fake-cpu"

    def prepare_detection_inputs(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Move each image tensor to the backend device (no reshape)."""
        return [img.to(self.device) for img in inputs]

    def __call__(self, inputs: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:  # type: ignore[override]
        self.received_inputs.append(inputs)
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 5.0, 5.0]]),
                "scores": torch.tensor([0.8]),
                "labels": torch.tensor([1], dtype=torch.int64),
            }
            for _ in inputs
        ]


class _FakeClassificationBackend(ModelBackend):
    """Minimal classification backend that echoes a fixed prediction tensor."""

    provides = frozenset({Capability.AUTOGRAD})

    def __init__(self, num_classes: int = 5) -> None:
        self.num_classes = num_classes
        self._task_kind = TaskKind.classification

    @property
    def task_kind(self) -> TaskKind:
        return self._task_kind

    @property
    def hardware_label(self) -> str:
        return "fake-cpu"

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        n = inputs.shape[0]
        return torch.zeros(n, self.num_classes)


# ---------------------------------------------------------------------------
# Detection forward pass tests
# ---------------------------------------------------------------------------


def test_detection_forward_ragged_list_returns_correct_length() -> None:
    """forward_pass with a ragged list[Tensor] produces detection_predictions of
    correct length and does not crash on shape[0] or torch.stack."""
    backend = _FakeDetectionBackend()
    config = _make_config(batch_size=32)

    # Two images at different native resolutions — cannot be stacked
    inputs: list[torch.Tensor] = [
        torch.zeros(3, 40, 50),
        torch.zeros(3, 60, 30),
    ]

    result = forward_pass(config, backend, inputs)

    assert result.task_kind is TaskKind.detection
    assert result.batch_size == 2
    assert len(result.as_detection()) == 2


def test_detection_forward_backend_receives_list_of_tensors() -> None:
    """The backend __call__ must receive a list[Tensor], not a stacked tensor."""
    backend = _FakeDetectionBackend()
    config = _make_config(batch_size=32)

    inputs: list[torch.Tensor] = [
        torch.zeros(3, 40, 50),
        torch.zeros(3, 60, 30),
    ]

    forward_pass(config, backend, inputs)

    assert len(backend.received_inputs) == 1  # one chunk (batch_size=32 > 2)
    chunk = backend.received_inputs[0]
    assert isinstance(chunk, list)
    assert all(isinstance(t, torch.Tensor) for t in chunk)


def test_detection_forward_chunked_across_batch_size() -> None:
    """With batch_size=1, the loop runs once per image; total predictions == num images."""
    backend = _FakeDetectionBackend()
    config = _make_config(batch_size=1)

    inputs: list[torch.Tensor] = [
        torch.zeros(3, 40, 50),
        torch.zeros(3, 60, 30),
        torch.zeros(3, 20, 80),
    ]

    result = forward_pass(config, backend, inputs)

    assert result.batch_size == 3
    assert len(result.as_detection()) == 3
    # Three separate chunks were passed to the model
    assert len(backend.received_inputs) == 3


def test_detection_forward_predictions_are_cpu_detached() -> None:
    """Each per-sample dict in detection_predictions must have CPU tensors."""
    backend = _FakeDetectionBackend()
    config = _make_config(batch_size=32)

    inputs: list[torch.Tensor] = [
        torch.zeros(3, 40, 50),
        torch.zeros(3, 60, 30),
    ]

    result = forward_pass(config, backend, inputs)

    for sample_dict in result.as_detection():
        for tensor in sample_dict.values():
            assert tensor.device.type == "cpu"


# ---------------------------------------------------------------------------
# Classification regression test
# ---------------------------------------------------------------------------


def test_classification_forward_dense_tensor_unchanged() -> None:
    """Classification branch: dense (N, C, H, W) tensor still works end-to-end."""
    backend = _FakeClassificationBackend(num_classes=10)
    config = _make_config(batch_size=32)

    inputs = torch.zeros(4, 3, 8, 8)

    result = forward_pass(config, backend, inputs)

    assert result.task_kind is TaskKind.classification
    assert result.batch_size == 4
    assert result.as_classification().shape == (4, 10)


def test_classification_forward_chunked_regression() -> None:
    """Classification branch: chunked path (total > batch_size) still works."""
    backend = _FakeClassificationBackend(num_classes=5)
    config = _make_config(batch_size=2)

    inputs = torch.zeros(6, 3, 8, 8)

    result = forward_pass(config, backend, inputs)

    assert result.task_kind is TaskKind.classification
    assert result.batch_size == 6
    assert result.as_classification().shape == (6, 5)


# ---------------------------------------------------------------------------
# Input-contract guard tests
# ---------------------------------------------------------------------------


def test_detection_forward_rejects_non_list_inputs() -> None:
    """A detection backend handed a dense tensor must fail loud, not slice it."""
    backend = _FakeDetectionBackend()
    config = _make_config()

    with pytest.raises(TypeError, match="expected a list"):
        forward_pass(config, backend, torch.zeros(2, 3, 8, 8))


def test_detection_forward_rejects_non_dict_backend_entries() -> None:
    """A detection backend returning list[non-dict] must fail loud at the entry, not .items()."""

    class _BadEntryBackend(_FakeDetectionBackend):
        def __call__(self, inputs: list[torch.Tensor]) -> Any:  # type: ignore[override]
            return [(0.9, "box") for _ in inputs]  # list, but tuples not dicts

    backend = _BadEntryBackend()
    config = _make_config()

    with pytest.raises(TypeError, match="dict of tensors"):
        forward_pass(config, backend, [torch.zeros(3, 8, 8)])


def test_classification_forward_rejects_list_inputs() -> None:
    """A classification backend handed a list must fail loud."""
    backend = _FakeClassificationBackend()
    config = _make_config()

    with pytest.raises(TypeError, match="expected a dense"):
        forward_pass(config, backend, [torch.zeros(3, 8, 8)])  # type: ignore[arg-type]


def test_classification_forward_rejects_unbatched_tensor() -> None:
    """A 1-D tensor lacks a sample axis; reject before len()/slicing."""
    backend = _FakeClassificationBackend()
    config = _make_config()

    with pytest.raises(ValueError, match="ndim >= 2"):
        forward_pass(config, backend, torch.zeros(5))


# ---------------------------------------------------------------------------
# Task family extract_forward tests
# ---------------------------------------------------------------------------


class _StubClsBackend:
    task_kind = TaskKind.classification

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 5)


def test_classification_family_extract_forward_returns_tensor() -> None:
    fam = resolve_task_family(TaskKind.classification)
    ctx = ForwardContext(
        backend=cast("ModelBackend", _StubClsBackend()), inputs=torch.zeros(3, 1, 4, 4)
    )
    payload = fam.extract_forward(ctx, batch_size=2)
    assert isinstance(payload, torch.Tensor)
    assert payload.shape[0] == 3
