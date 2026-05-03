from __future__ import annotations

import ast
import importlib.util
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, get_type_hints


def _load_contracts_module() -> Any:
    module_name = "_raitap_transparency_contracts_under_test"
    module_path = Path(__file__).resolve().parents[1] / "contracts.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _public_names_from_init() -> set[str]:
    init_path = Path(__file__).resolve().parents[1] / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise AssertionError("__all__ assignment not found")


contracts: Any = _load_contracts_module()

ExplanationOutputSpace = contracts.ExplanationOutputSpace
ExplanationPayloadKind = contracts.ExplanationPayloadKind
ExplanationScope = contracts.ExplanationScope
ExplanationSemantics = contracts.ExplanationSemantics
ExplanationTarget = contracts.ExplanationTarget
InputSpec = contracts.InputSpec
MethodFamily = contracts.MethodFamily
OutputSpaceSpec = contracts.OutputSpaceSpec
SampleSelection = contracts.SampleSelection
ScopeDefinitionStep = contracts.ScopeDefinitionStep


def test_semantic_enum_members_are_exact_and_not_placeholders() -> None:
    assert {member.name for member in ExplanationScope} == {"LOCAL", "COHORT", "GLOBAL"}
    assert {member.name for member in ScopeDefinitionStep} == {
        "EXPLAINER_OUTPUT",
        "VISUALISER_SUMMARY",
    }
    assert {member.name for member in ExplanationOutputSpace} == {
        "INPUT_FEATURES",
        "INTERPRETABLE_FEATURES",
        "LAYER_ACTIVATION",
        "IMAGE_SPATIAL_MAP",
        "TOKEN_SEQUENCE",
    }
    assert {member.name for member in MethodFamily} == {
        "GRADIENT",
        "PERTURBATION",
        "SHAPLEY",
        "CAM",
        "MODEL_AGNOSTIC",
        "TREE",
        "SURROGATE",
    }

    placeholders = {"UNKNOWN", "UNSPECIFIED", "UNCLASSIFIED", "PLACEHOLDER"}
    semantic_enums = (
        ExplanationScope,
        ScopeDefinitionStep,
        ExplanationOutputSpace,
        MethodFamily,
    )
    for enum_cls in semantic_enums:
        for member in enum_cls:
            assert member.name.upper() not in placeholders
            assert member.value.upper() not in placeholders


def test_contract_does_not_define_explanation_unit() -> None:
    assert not hasattr(contracts, "ExplanationUnit")
    assert "ExplanationUnit" not in _public_names_from_init()


def test_explanation_semantics_has_only_contract_fields() -> None:
    assert [field.name for field in fields(ExplanationSemantics)] == [
        "scope",
        "scope_definition_step",
        "payload_kind",
        "method_families",
        "target",
        "sample_selection",
        "input_spec",
        "output_space",
    ]
    assert "primary_method_family" not in {field.name for field in fields(ExplanationSemantics)}


def test_public_contract_type_hints_resolve_without_optional_runtime_imports() -> None:
    assert get_type_hints(InputSpec)["metadata"] == contracts.Mapping[str, Any] | None

    explain_hints = get_type_hints(contracts.ExplainerAdapter.explain)
    assert explain_hints["model"] is Any
    assert explain_hints["inputs"] is Any
    assert explain_hints["return"] is Any


def test_sample_ids_and_display_names_remain_separate() -> None:
    sample_selection = SampleSelection(
        sample_ids=["stable-1", "stable-2"],
        sample_display_names=["First row", "Second row"],
    )
    semantics = ExplanationSemantics(
        scope=ExplanationScope.LOCAL,
        scope_definition_step=ScopeDefinitionStep.EXPLAINER_OUTPUT,
        payload_kind=ExplanationPayloadKind.ATTRIBUTIONS,
        method_families=frozenset({MethodFamily.GRADIENT}),
        target=ExplanationTarget(target=1, label="positive"),
        sample_selection=sample_selection,
        input_spec=InputSpec(
            kind="tabular",
            shape=(2, 3),
            layout="(B,F)",
            feature_names=["a", "b", "c"],
        ),
        output_space=OutputSpaceSpec(
            space=ExplanationOutputSpace.INPUT_FEATURES,
            shape=(2, 3),
            layout="(B,F)",
            feature_names=["a", "b", "c"],
        ),
    )

    assert semantics.sample_selection is not None
    assert semantics.sample_selection.sample_ids == ["stable-1", "stable-2"]
    assert semantics.sample_selection.sample_display_names == ["First row", "Second row"]
    assert semantics.sample_selection.sample_ids != semantics.sample_selection.sample_display_names


def test_public_transparency_exports_new_contract_surface_without_report_scope() -> None:
    public_names = _public_names_from_init()
    assert {
        "ExplainerCapability",
        "ExplanationOutputSpace",
        "ExplanationScope",
        "ExplanationSemantics",
        "ExplanationTarget",
        "InputSpec",
        "MethodFamily",
        "OutputSpaceSpec",
        "SampleSelection",
        "ScopeDefinitionStep",
        "VisualSummarySpec",
        "explainer_capability",
        "infer_input_spec",
        "infer_output_space",
        "method_families_for_explainer",
    } <= public_names
    assert "report_scope" not in public_names
