from __future__ import annotations

from typing import Any

import torch

from raitap.pipeline.outputs import ForwardOutput, OutputKind
from raitap.pipeline.phases.forward_pass import forward_pass
from raitap.testing import make_app_config, make_fake_backend
from raitap.types import Capability, TaskKind


def _config() -> Any:  # a faithful AppConfig stand-in for forward_pass
    return make_app_config()


def test_default_output_kind_is_logits() -> None:
    out = ForwardOutput(task_kind=TaskKind.classification, batch_size=1, payload=torch.zeros(1, 2))
    assert out.output_kind == OutputKind.LOGITS


def test_forward_pass_stamps_probabilities_for_predict_proba_backend() -> None:
    backend = make_fake_backend(
        provides=frozenset({Capability.TREE_MODEL, Capability.PREDICT_PROBA})
    )
    out = forward_pass(_config(), backend, torch.zeros(4, 3))
    assert out.output_kind == OutputKind.PROBABILITIES


def test_forward_pass_stamps_logits_for_autograd_backend() -> None:
    backend = make_fake_backend(provides=frozenset({Capability.AUTOGRAD}))
    out = forward_pass(_config(), backend, torch.zeros(4, 3))
    assert out.output_kind == OutputKind.LOGITS
