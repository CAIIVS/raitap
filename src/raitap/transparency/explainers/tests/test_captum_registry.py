"""Tests for captum Layer* registry entries (#267)."""

from __future__ import annotations

import pytest

from raitap.transparency.contracts import BaselineCardinality, BaselineMode, MethodFamily
from raitap.transparency.explainers.captum_explainer import CaptumExplainer

_LAYER_METHODS = [
    "LayerConductance",
    "LayerIntegratedGradients",
    "LayerActivation",
    "LayerDeepLift",
    "LayerGradientXActivation",
    "LayerLRP",
]


@pytest.mark.parametrize("algorithm", _LAYER_METHODS)
def test_layer_methods_registered_with_gradient_family(algorithm: str) -> None:
    registry = CaptumExplainer.algorithm_registry
    assert algorithm in registry
    assert MethodFamily.GRADIENT in registry[algorithm].families
    assert registry[algorithm].stochastic is False


@pytest.mark.parametrize(
    ("algorithm", "expected_mode", "expected_cardinality"),
    [
        # Integral Layer* methods fall back to a zero reference in captum.
        ("LayerConductance", BaselineMode.ZERO, BaselineCardinality.SINGLE),
        ("LayerIntegratedGradients", BaselineMode.ZERO, BaselineCardinality.SINGLE),
        ("LayerDeepLift", BaselineMode.ZERO, BaselineCardinality.SINGLE),
        # Activation / gradient-product methods take no reference input.
        ("LayerActivation", None, None),
        ("LayerGradientXActivation", None, None),
        ("LayerLRP", None, None),
    ],
)
def test_layer_method_baseline_hints(
    algorithm: str,
    expected_mode: BaselineMode | None,
    expected_cardinality: BaselineCardinality | None,
) -> None:
    hints = CaptumExplainer.algorithm_registry[algorithm]
    assert hints.baseline_default == expected_mode
    assert hints.baseline_cardinality == expected_cardinality


def test_every_captum_stochastic_entry_is_global_rng() -> None:
    for name, spec in CaptumExplainer.algorithm_registry.items():
        assert spec.seeding in {"deterministic", "global_rng"}, name
