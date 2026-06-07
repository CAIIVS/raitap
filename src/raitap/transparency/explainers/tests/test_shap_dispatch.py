"""SHAP dispatch via per-entry invoker (#266)."""

from __future__ import annotations

import pytest

from raitap.transparency.contracts import MethodFamily
from raitap.transparency.explainers.shap_explainer import (
    ShapExplainer,
    _shap_legacy_invoker,
    _shap_modern_invoker,
)


@pytest.mark.parametrize(
    ("algorithm", "invoker"),
    [
        ("GradientExplainer", _shap_legacy_invoker),
        ("KernelExplainer", _shap_legacy_invoker),
        ("SamplingExplainer", _shap_legacy_invoker),
        ("PartitionExplainer", _shap_modern_invoker),
        ("ExactExplainer", _shap_modern_invoker),
        ("PermutationExplainer", _shap_modern_invoker),
    ],
)
def test_entry_carries_expected_invoker(algorithm: str, invoker: object) -> None:
    assert ShapExplainer.algorithm_registry[algorithm].invoker is invoker


def test_every_entry_uses_correct_invoker() -> None:
    # Drift guard: modern algorithms use the modern invoker, all others the legacy one.
    modern = {"PartitionExplainer", "ExactExplainer", "PermutationExplainer"}
    for name, hints in ShapExplainer.algorithm_registry.items():
        expected = _shap_modern_invoker if name in modern else _shap_legacy_invoker
        assert hints.invoker is expected, name


@pytest.mark.parametrize(
    ("algorithm", "stochastic"),
    [
        ("PartitionExplainer", False),
        ("ExactExplainer", False),
        ("PermutationExplainer", True),
        ("SamplingExplainer", True),
    ],
)
def test_modern_and_sampling_entry_metadata(algorithm: str, stochastic: bool) -> None:
    hints = ShapExplainer.algorithm_registry[algorithm]
    assert hints.stochastic is stochastic
    expected = {MethodFamily.SHAPLEY, MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC}
    assert expected <= hints.families
