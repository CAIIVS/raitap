"""Transparency stochastic resolution + registry flags (issue #251)."""

from __future__ import annotations

from types import SimpleNamespace

from raitap.transparency.explainers.captum_explainer import CaptumExplainer
from raitap.transparency.explainers.shap_explainer import ShapExplainer
from raitap.transparency.semantics import explainer_is_stochastic


def test_explainer_is_stochastic_reads_registry() -> None:
    assert explainer_is_stochastic(CaptumExplainer(algorithm="KernelShap")) is True
    assert explainer_is_stochastic(CaptumExplainer(algorithm="IntegratedGradients")) is False
    assert explainer_is_stochastic(ShapExplainer(algorithm="GradientExplainer")) is True
    assert explainer_is_stochastic(ShapExplainer(algorithm="DeepExplainer")) is False


def test_explainer_is_stochastic_defaults_false_for_unknown() -> None:
    stub = SimpleNamespace(algorithm="<unknown>")
    assert explainer_is_stochastic(stub) is False


def test_registry_stochastic_flags() -> None:
    assert ShapExplainer.algorithm_registry["GradientExplainer"].stochastic is True
    assert ShapExplainer.algorithm_registry["KernelExplainer"].stochastic is True
    assert ShapExplainer.algorithm_registry["DeepExplainer"].stochastic is False
    assert ShapExplainer.algorithm_registry["TreeExplainer"].stochastic is False

    assert CaptumExplainer.algorithm_registry["ShapleyValueSampling"].stochastic is True
    assert CaptumExplainer.algorithm_registry["Lime"].stochastic is True
    assert CaptumExplainer.algorithm_registry["IntegratedGradients"].stochastic is False
    assert CaptumExplainer.algorithm_registry["LayerGradCam"].stochastic is False
