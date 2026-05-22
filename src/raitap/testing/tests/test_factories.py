from __future__ import annotations

import torch

from raitap.testing import make_app_config, make_tiny_classifier


def test_tiny_classifier_is_deterministic_and_runs() -> None:
    a = make_tiny_classifier(seed=0)
    b = make_tiny_classifier(seed=0)
    x = torch.zeros(1, 3, 8, 8)
    assert torch.equal(a(x), b(x))


def test_app_config_factory_overrides() -> None:
    cfg = make_app_config(experiment_name="demo", hardware="cpu")
    assert cfg.experiment_name == "demo"
    assert cfg.hardware == "cpu"
