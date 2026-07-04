"""Transparency seeding resolution + registry flags (issues #251, #339)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from raitap.transparency.explainers.captum_explainer import CaptumExplainer
from raitap.transparency.explainers.shap_explainer import ShapExplainer
from raitap.transparency.semantics import explainer_seeding


def test_explainer_seeding_reads_registry() -> None:
    assert explainer_seeding(CaptumExplainer(algorithm="KernelShap")) == "global_rng"
    assert explainer_seeding(CaptumExplainer(algorithm="IntegratedGradients")) == "deterministic"
    assert explainer_seeding(ShapExplainer(algorithm="GradientExplainer")) == "global_rng"
    assert explainer_seeding(ShapExplainer(algorithm="DeepExplainer")) == "deterministic"


def test_explainer_seeding_defaults_deterministic_for_unknown() -> None:
    stub = SimpleNamespace(algorithm="<unknown>")
    assert explainer_seeding(stub) == "deterministic"


def test_registry_stochastic_flags() -> None:
    assert ShapExplainer.algorithm_registry["GradientExplainer"].stochastic is True
    assert ShapExplainer.algorithm_registry["KernelExplainer"].stochastic is True
    assert ShapExplainer.algorithm_registry["DeepExplainer"].stochastic is False
    assert ShapExplainer.algorithm_registry["TreeExplainer"].stochastic is False

    assert CaptumExplainer.algorithm_registry["ShapleyValueSampling"].stochastic is True
    assert CaptumExplainer.algorithm_registry["Lime"].stochastic is True
    assert CaptumExplainer.algorithm_registry["IntegratedGradients"].stochastic is False
    assert CaptumExplainer.algorithm_registry["LayerGradCam"].stochastic is False


@pytest.mark.parametrize(
    ("algorithm", "expected"),
    [
        ("PermutationExplainer", "self_seeded"),
        ("SamplingExplainer", "global_rng"),
        ("PartitionExplainer", "deterministic"),
        ("ExactExplainer", "deterministic"),
    ],
)
def test_modern_shap_seeding(algorithm: str, expected: str) -> None:
    assert explainer_seeding(ShapExplainer(algorithm=algorithm)) == expected


def test_explanation_semantics_carries_seeding() -> None:
    from raitap.transparency.contracts import (
        ExplanationOutputSpace,
        ExplanationPayloadKind,
        ExplanationScope,
        ExplanationSemantics,
        OutputSpaceSpec,
        ScopeDefinitionStep,
    )

    sem = ExplanationSemantics(
        scope=ExplanationScope.LOCAL,
        scope_definition_step=ScopeDefinitionStep.EXPLAINER_OUTPUT,
        payload_kind=ExplanationPayloadKind.ATTRIBUTIONS,
        method_families=frozenset(),
        target=None,
        sample_selection=None,
        input_spec=None,
        output_space=OutputSpaceSpec(
            space=ExplanationOutputSpace.INPUT_FEATURES,
            shape=(1,),
            layout=None,
        ),
        seeding="self_seeded",
    )
    assert sem.seeding == "self_seeded"
    assert sem.stochastic is True
