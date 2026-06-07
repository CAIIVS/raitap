"""SHAP API dispatch table + modern/legacy registry entries (#267)."""

from __future__ import annotations

import pytest

from raitap.transparency.contracts import MethodFamily
from raitap.transparency.explainers.shap_explainer import _SHAP_API, ShapExplainer


@pytest.mark.parametrize(
    ("algorithm", "api"),
    [
        ("GradientExplainer", "legacy"),
        ("KernelExplainer", "legacy"),
        ("SamplingExplainer", "legacy"),
        ("PartitionExplainer", "modern"),
        ("ExactExplainer", "modern"),
        ("PermutationExplainer", "modern"),
    ],
)
def test_dispatch_table(algorithm: str, api: str) -> None:
    assert _SHAP_API[algorithm] == api


@pytest.mark.parametrize(
    ("algorithm", "stochastic"),
    [
        ("PartitionExplainer", False),
        ("ExactExplainer", False),
        ("PermutationExplainer", True),
        ("SamplingExplainer", True),
    ],
)
def test_new_registry_entries(algorithm: str, stochastic: bool) -> None:
    hints = ShapExplainer.algorithm_registry[algorithm]
    assert hints.stochastic is stochastic
    expected = {MethodFamily.SHAPLEY, MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC}
    assert expected <= hints.families
