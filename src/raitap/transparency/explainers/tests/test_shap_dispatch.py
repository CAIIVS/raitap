"""SHAP API dispatch via registry entries + modern/legacy split (#267)."""

from __future__ import annotations

import pytest

from raitap.transparency.contracts import MethodFamily
from raitap.transparency.explainers.shap_explainer import ShapExplainer, ShapExplainerHints


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
def test_registry_carries_api(algorithm: str, api: str) -> None:
    hints = ShapExplainer.algorithm_registry[algorithm]
    assert isinstance(hints, ShapExplainerHints)  # narrows for the .api read
    assert hints.api == api


def test_every_shap_entry_declares_api() -> None:
    # Drift guard: a new explainer added without an api would not be ShapExplainerHints.
    for algorithm, hints in ShapExplainer.algorithm_registry.items():
        assert isinstance(hints, ShapExplainerHints), algorithm


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
