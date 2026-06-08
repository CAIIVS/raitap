from __future__ import annotations

from types import SimpleNamespace

import torch

from raitap.pipeline.outputs import ForwardOutput, OutputKind
from raitap.pipeline.phases.forward_pass import forward_pass
from raitap.types import Capability, TaskKind


class _ProbaBackend:
    provides = frozenset({Capability.TREE_MODEL, Capability.PREDICT_PROBA})
    task_kind = TaskKind.classification
    expected_input_shape = None

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        n = inputs.shape[0]
        return torch.full((n, 2), 0.5)


class _LogitBackend(_ProbaBackend):
    provides = frozenset({Capability.AUTOGRAD})


def _config() -> object:
    return SimpleNamespace(run=SimpleNamespace(forward_batch_size=8), data=None)


def test_default_output_kind_is_logits() -> None:
    out = ForwardOutput(task_kind=TaskKind.classification, batch_size=1, payload=torch.zeros(1, 2))
    assert out.output_kind == OutputKind.LOGITS


def test_forward_pass_stamps_probabilities_for_predict_proba_backend() -> None:
    out = forward_pass(_config(), _ProbaBackend(), torch.zeros(4, 3))
    assert out.output_kind == OutputKind.PROBABILITIES


def test_forward_pass_stamps_logits_for_autograd_backend() -> None:
    out = forward_pass(_config(), _LogitBackend(), torch.zeros(4, 3))
    assert out.output_kind == OutputKind.LOGITS
