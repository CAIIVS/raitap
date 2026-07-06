from __future__ import annotations

import pytest
import torch
from omegaconf import DictConfig

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


def test_make_app_config_returns_struct_dictconfig_with_schema_defaults() -> None:
    cfg = make_app_config()
    assert isinstance(cfg, DictConfig)
    # Declared defaults are present without any override.
    assert cfg.transparency == {}
    assert cfg.robustness == {}
    assert cfg.metrics is None
    assert cfg.reporting is None
    assert cfg.tracking is None
    assert cfg.seed is None
    assert str(cfg.hardware) == "gpu"
    assert cfg.experiment_name == "Experiment"


def test_make_app_config_applies_flat_and_nested_overrides() -> None:
    cfg = make_app_config(experiment_name="demo", hardware="cpu", model={"source": "resnet18"})
    assert cfg.experiment_name == "demo"
    assert str(cfg.hardware) == "cpu"
    assert cfg.model.source == "resnet18"
    # Untouched nested defaults survive the merge.
    assert cfg.model.pretrained is False


def test_make_app_config_reading_undeclared_field_raises() -> None:
    cfg = make_app_config()
    with pytest.raises(AttributeError):
        _ = cfg.definitely_not_a_field
