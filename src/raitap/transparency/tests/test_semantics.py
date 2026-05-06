from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def _load_contracts_and_semantics() -> tuple[Any, Any]:
    module_root = Path(__file__).resolve().parents[1]
    package_name = "_raitap_transparency_semantics_under_test"
    package = ModuleType(package_name)
    package.__path__ = [str(module_root)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package

    contracts = _load_module(
        f"{package_name}.contracts",
        module_root / "contracts.py",
    )
    semantics = _load_module(
        f"{package_name}.semantics",
        module_root / "semantics.py",
    )
    return contracts, semantics


def _load_module(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


contracts, semantics = _load_contracts_and_semantics()

ExplanationOutputSpace = contracts.ExplanationOutputSpace
ExplanationPayloadKind = contracts.ExplanationPayloadKind
ExplanationScope = contracts.ExplanationScope
InputKind = contracts.InputKind
InputSpec = contracts.InputSpec
MethodFamily = contracts.MethodFamily
ScopeDefinitionStep = contracts.ScopeDefinitionStep
TensorLayout = contracts.TensorLayout
CAPTUM_METHOD_FAMILIES = semantics.CAPTUM_METHOD_FAMILIES
SHAP_METHOD_FAMILIES = semantics.SHAP_METHOD_FAMILIES
explainer_capability = semantics.explainer_capability
infer_input_spec = semantics.infer_input_spec
infer_output_space = semantics.infer_output_space
method_families_for_explainer = semantics.method_families_for_explainer


def _explainer(framework: str, algorithm: str) -> object:
    return SimpleNamespace(framework=framework, algorithm=algorithm)


def test_shap_method_family_registry_accepts_adr_v1_list() -> None:
    assert set(SHAP_METHOD_FAMILIES) == {
        "GradientExplainer",
        "DeepExplainer",
        "KernelExplainer",
        "TreeExplainer",
    }
    assert method_families_for_explainer(_explainer("shap", "GradientExplainer")) == frozenset(
        {MethodFamily.SHAPLEY, MethodFamily.GRADIENT}
    )
    assert method_families_for_explainer(_explainer("shap", "DeepExplainer")) == frozenset(
        {MethodFamily.SHAPLEY, MethodFamily.GRADIENT}
    )
    assert method_families_for_explainer(_explainer("shap", "KernelExplainer")) == frozenset(
        {MethodFamily.SHAPLEY, MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC}
    )
    assert method_families_for_explainer(_explainer("shap", "TreeExplainer")) == frozenset(
        {MethodFamily.SHAPLEY, MethodFamily.TREE}
    )


def test_shap_permutation_explainer_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "method-family inference is not implemented for framework "
            "SHAP and algorithm PermutationExplainer"
        ),
    ):
        method_families_for_explainer(_explainer("shap", "PermutationExplainer"))


def test_captum_method_family_registry_exactly_matches_adr_v1_list() -> None:
    expected = {
        "IntegratedGradients": frozenset({MethodFamily.GRADIENT}),
        "Saliency": frozenset({MethodFamily.GRADIENT}),
        "FeatureAblation": frozenset({MethodFamily.PERTURBATION}),
        "FeaturePermutation": frozenset({MethodFamily.PERTURBATION}),
        "Occlusion": frozenset({MethodFamily.PERTURBATION}),
        "ShapleyValueSampling": frozenset({MethodFamily.SHAPLEY, MethodFamily.PERTURBATION}),
        "ShapleyValues": frozenset({MethodFamily.SHAPLEY, MethodFamily.PERTURBATION}),
        "KernelShap": frozenset(
            {MethodFamily.SHAPLEY, MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC}
        ),
        "Lime": frozenset(
            {MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC, MethodFamily.SURROGATE}
        ),
        "LayerGradCam": frozenset({MethodFamily.GRADIENT, MethodFamily.CAM}),
        "GuidedGradCam": frozenset({MethodFamily.GRADIENT, MethodFamily.CAM}),
    }

    assert expected == CAPTUM_METHOD_FAMILIES
    for algorithm, families in expected.items():
        assert method_families_for_explainer(_explainer("captum", algorithm)) == families


def test_bare_captum_grad_cam_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "method-family inference is not implemented for framework Captum and algorithm GradCam"
        ),
    ):
        method_families_for_explainer(_explainer("captum", "GradCam"))


def test_explainer_capability_uses_local_explainer_output_contract() -> None:
    capability = explainer_capability(_explainer("captum", "LayerGradCam"))

    assert capability.scope is ExplanationScope.LOCAL
    assert capability.scope_definition_step is ScopeDefinitionStep.EXPLAINER_OUTPUT
    assert capability.payload_kind is ExplanationPayloadKind.ATTRIBUTIONS
    assert capability.method_families == frozenset({MethodFamily.GRADIENT, MethodFamily.CAM})
    assert capability.candidate_output_spaces == frozenset(
        {
            ExplanationOutputSpace.IMAGE_SPATIAL_MAP,
            ExplanationOutputSpace.LAYER_ACTIVATION,
        }
    )


def test_explainer_capability_uses_explainer_declared_output_scope() -> None:
    class GlobalScopeExplainer:
        framework = "captum"
        algorithm = "LayerGradCam"
        output_scope = ExplanationScope.GLOBAL

    capability = explainer_capability(GlobalScopeExplainer())

    assert capability.scope is ExplanationScope.GLOBAL


def test_explainer_capability_defaults_to_local_scope_without_declaration() -> None:
    capability = explainer_capability(_explainer("captum", "LayerGradCam"))

    assert capability.scope is ExplanationScope.LOCAL


def test_infer_input_spec_preserves_explicit_metadata() -> None:
    metadata = {"kind": "tabular", "shape": (4, 3), "layout": "(B,F)"}

    input_spec = infer_input_spec(input_metadata=metadata, feature_names=["x", "y", "z"])

    assert input_spec == InputSpec(
        kind=InputKind.TABULAR,
        shape=(4, 3),
        layout=TensorLayout.BATCH_FEATURE,
        feature_names=["x", "y", "z"],
        metadata=metadata,
    )
    assert input_spec.kind is InputKind.TABULAR
    assert input_spec.layout is TensorLayout.BATCH_FEATURE


@pytest.mark.parametrize(
    ("raw_layout", "expected_layout"),
    [
        ("B,F", TensorLayout.BATCH_FEATURE),
        ("(B,F)", TensorLayout.BATCH_FEATURE),
        ("B,T,C", TensorLayout.BATCH_TIME_CHANNEL),
        ("(B,T,C)", TensorLayout.BATCH_TIME_CHANNEL),
        ("TOKEN_SEQUENCE", TensorLayout.TOKEN_SEQUENCE),
        ("TOKENS", TensorLayout.TOKEN_SEQUENCE),
    ],
)
def test_infer_input_spec_accepts_legacy_layout_aliases(
    raw_layout: str,
    expected_layout: object,
) -> None:
    input_spec = infer_input_spec(input_metadata={"kind": "text", "layout": raw_layout})

    assert input_spec.layout is expected_layout


def test_infer_output_space_requires_metadata_for_ambiguous_shapes() -> None:
    with pytest.raises(ValueError, match="shape alone is ambiguous"):
        infer_output_space(input_spec=InputSpec(kind=None, shape=(4, 3), layout=None))


def test_infer_output_space_error_names_config_key_and_valid_kinds() -> None:
    with pytest.raises(ValueError) as exc_info:
        infer_output_space(input_spec=InputSpec(kind=None, shape=(4, 3), layout=None))
    msg = str(exc_info.value)
    # Names both keys explicitly so users know they have a choice.
    assert "input_metadata" in msg
    assert "kind" in msg
    assert "layout" in msg
    # Lists every valid kind.
    for kind in ("image", "tabular", "text", "time_series"):
        assert kind in msg
    # Lists every valid layout.
    for layout in ("NCHW", "(B,F)", "(B,T,C)", "TOKENS"):
        assert layout in msg
    # Includes a YAML example so users can copy-paste the fix.
    assert "raitap:" in msg
    assert "input_metadata:" in msg


def test_infer_output_space_layout_alone_disambiguates() -> None:
    # Setting only ``layout`` (no ``kind``) is enough — covers what the
    # error message promises about either key being sufficient.
    spec = InputSpec(kind=None, shape=(2, 3, 8, 8), layout="NCHW")
    output_space = infer_output_space(input_spec=spec, algorithm="IntegratedGradients")
    assert output_space.space is ExplanationOutputSpace.INPUT_FEATURES


def test_infer_output_space_uses_cam_method_family_for_image_spatial_maps() -> None:
    output_space = infer_output_space(
        input_spec=InputSpec(kind="image", shape=(2, 3, 8, 8), layout="NCHW"),
        method_families=frozenset({MethodFamily.GRADIENT, MethodFamily.CAM}),
        layer_path="features.0",
    )

    assert output_space.space is ExplanationOutputSpace.IMAGE_SPATIAL_MAP
    assert output_space.layout is TensorLayout.BATCH_CHANNEL_HEIGHT_WIDTH
    assert output_space.layer_path == "features.0"
    assert output_space.requires_interpolation is True


def test_infer_output_space_uses_algorithm_only_cam_signal() -> None:
    output_space = infer_output_space(
        input_spec=InputSpec(kind="image", shape=(2, 3, 8, 8), layout="NCHW"),
        algorithm="LayerGradCam",
        layer_path="features.0",
    )

    assert output_space.space is ExplanationOutputSpace.IMAGE_SPATIAL_MAP
    assert output_space.requires_interpolation is True


def test_infer_output_space_explicit_algorithm_takes_precedence_over_explainer() -> None:
    output_space = infer_output_space(
        input_spec=InputSpec(kind="image", shape=(2, 3, 8, 8), layout="NCHW"),
        explainer=_explainer("captum", "LayerGradCam"),
        algorithm="IntegratedGradients",
    )

    assert output_space.space is ExplanationOutputSpace.INPUT_FEATURES
    assert output_space.requires_interpolation is False


def test_infer_output_space_rejects_unknown_algorithm_only_signal() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "method-family inference is not implemented for framework "
            "<unknown> and algorithm MadeUpExplainer"
        ),
    ):
        infer_output_space(
            input_spec=InputSpec(kind="image", shape=(2, 3, 8, 8), layout="NCHW"),
            algorithm="MadeUpExplainer",
        )


def test_infer_output_space_rejects_ambiguous_algorithm_only_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        SHAP_METHOD_FAMILIES,
        "SharedAlgorithm",
        frozenset({MethodFamily.SHAPLEY}),
    )
    monkeypatch.setitem(
        CAPTUM_METHOD_FAMILIES,
        "SharedAlgorithm",
        frozenset({MethodFamily.GRADIENT}),
    )

    with pytest.raises(
        ValueError,
        match=(
            "method-family inference is ambiguous for algorithm "
            "SharedAlgorithm; matched frameworks SHAP, Captum"
        ),
    ):
        infer_output_space(
            input_spec=InputSpec(kind="image", shape=(2, 3, 8, 8), layout="NCHW"),
            algorithm="SharedAlgorithm",
        )
