from __future__ import annotations

import ast
import dataclasses
import importlib.util
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, get_type_hints

import pytest


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
InputKind = contracts.InputKind
InputSpec = contracts.InputSpec
MethodFamily = contracts.MethodFamily
OutputSpaceSpec = contracts.OutputSpaceSpec
SampleSelection = contracts.SampleSelection
ScopeDefinitionStep = contracts.ScopeDefinitionStep
TensorLayout = contracts.TensorLayout
explainer_output_scope = contracts.explainer_output_scope


def test_semantic_enum_members_are_exact_and_not_placeholders() -> None:
    assert {member.name for member in ExplanationScope} == {"LOCAL", "AGGREGATED", "GLOBAL"}
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
        "DETECTION_BOXES",
        "SEGMENTATION_MASK",
        "BBOX_REGRESSION",
    }
    assert {member.name for member in InputKind} == {
        "IMAGE",
        "TABULAR",
        "TEXT",
        "TIME_SERIES",
    }
    assert {member.name for member in TensorLayout} == {
        "BATCH_CHANNEL_HEIGHT_WIDTH",
        "BATCH_FEATURE",
        "BATCH_TIME_CHANNEL",
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
        InputKind,
        TensorLayout,
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
        "seeding",
    ]
    assert "primary_method_family" not in {field.name for field in fields(ExplanationSemantics)}


def test_public_contract_type_hints_resolve_without_optional_runtime_imports() -> None:
    assert get_type_hints(InputSpec)["kind"] == InputKind | None
    assert get_type_hints(InputSpec)["layout"] == TensorLayout | None
    assert get_type_hints(InputSpec)["metadata"] == contracts.Mapping[str, Any] | None
    assert get_type_hints(InputSpec.__init__)["kind"] == InputKind | str | None
    assert get_type_hints(InputSpec.__init__)["layout"] == TensorLayout | str | None
    assert get_type_hints(OutputSpaceSpec)["layout"] == TensorLayout | None
    assert get_type_hints(OutputSpaceSpec.__init__)["layout"] == TensorLayout | str | None

    explain_hints = get_type_hints(contracts.ExplainerAdapter.explain)
    assert explain_hints["model"] is Any
    assert explain_hints["inputs"] is Any
    assert explain_hints["return"] is Any


def test_input_spec_normalises_legacy_strings_to_typed_metadata() -> None:
    input_spec = InputSpec(kind="image", shape=(2, 3, 8, 8), layout="NCHW")

    assert input_spec.kind is InputKind.IMAGE
    assert input_spec.layout is TensorLayout.BATCH_CHANNEL_HEIGHT_WIDTH


def test_output_space_spec_normalises_layout_aliases() -> None:
    output_space = OutputSpaceSpec(
        space=ExplanationOutputSpace.INPUT_FEATURES,
        shape=(2, 3),
        layout="B,F",
    )

    assert output_space.layout is TensorLayout.BATCH_FEATURE


def test_explainer_output_scope_uses_declared_class_contract_or_local_default() -> None:
    class GlobalScopeExplainer:
        output_scope = ExplanationScope.GLOBAL

    assert explainer_output_scope(GlobalScopeExplainer()) is ExplanationScope.GLOBAL
    assert explainer_output_scope(object()) is ExplanationScope.LOCAL


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


def test_explanation_output_space_includes_detection_members() -> None:
    values = {member.value for member in ExplanationOutputSpace}
    assert "detection_boxes" in values
    assert "segmentation_mask" in values
    assert "bbox_regression" in values


def test_detection_box_round_trip() -> None:
    from raitap.transparency.contracts import DetectionBox

    box = DetectionBox(
        display_index=0,
        raw_index=7,
        xyxy=(1.0, 2.0, 3.0, 4.0),
        score=0.93,
        label_index=5,
        label_name="car",
    )
    assert box.display_index == 0
    assert box.raw_index == 7
    assert box.xyxy == (1.0, 2.0, 3.0, 4.0)
    assert box.score == 0.93
    assert box.label_index == 5
    assert box.label_name == "car"


def test_detection_box_label_name_defaults_to_none() -> None:
    from raitap.transparency.contracts import DetectionBox

    box = DetectionBox(
        display_index=0, raw_index=0, xyxy=(0.0, 0.0, 1.0, 1.0), score=0.5, label_index=1
    )
    assert box.label_name is None


def test_detection_box_is_frozen() -> None:
    from raitap.transparency.contracts import DetectionBox

    box = DetectionBox(
        display_index=0, raw_index=0, xyxy=(0.0, 0.0, 1.0, 1.0), score=0.5, label_index=1
    )
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        box.score = 0.9  # type: ignore[misc]


def test_visualisation_context_detection_box_defaults_to_none() -> None:
    from raitap.transparency.contracts import VisualisationContext

    ctx = VisualisationContext(algorithm="x", sample_names=None, show_sample_names=False)
    assert ctx.detection_box is None


def test_baseline_record_is_frozen_and_carries_descriptor() -> None:
    import dataclasses
    from pathlib import Path

    from raitap.transparency.contracts import BaselineRecord

    record = BaselineRecord(
        kwarg_name="background_data",
        mode="configured",
        source="imagenet_samples",
        n_samples=50,
        shape=(50, 3, 224, 224),
        dtype="torch.float32",
        sha256="abc123",
        image_path=Path("baseline.png"),
    )
    assert record.kwarg_name == "background_data"
    assert record.mode == "configured"
    assert record.shape == (50, 3, 224, 224)
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.mode = "zero"  # type: ignore[misc]  # frozen


def test_detection_box_gt_fields_default_unset() -> None:
    from raitap.transparency.contracts import DetectionBox

    box = DetectionBox(display_index=0, raw_index=0, xyxy=(0, 0, 1, 1), score=0.9, label_index=38)
    assert box.true_label_index is None
    assert box.true_label_name is None
    assert box.true_match_iou is None
    assert box.ground_truth_evaluated is False


def test_explainer_spec_seeding_drives_stochastic_property() -> None:
    from raitap.transparency.contracts import ExplainerAlgorithmSpec

    det = ExplainerAlgorithmSpec(families=frozenset())
    assert det.seeding == "deterministic"
    assert det.stochastic is False

    glob = ExplainerAlgorithmSpec(families=frozenset(), seeding="global_rng")
    assert glob.stochastic is True

    self_seeded = ExplainerAlgorithmSpec(families=frozenset(), seeding="self_seeded")
    assert self_seeded.stochastic is True


def test_detection_box_gt_fields_set() -> None:
    from raitap.transparency.contracts import DetectionBox

    box = DetectionBox(
        display_index=0,
        raw_index=0,
        xyxy=(0, 0, 1, 1),
        score=0.9,
        label_index=38,
        label_name="kite",
        true_label_index=20,
        true_label_name="sheep",
        true_match_iou=0.71,
        ground_truth_evaluated=True,
    )
    assert box.true_label_name == "sheep"
    assert box.true_match_iou == 0.71
    assert box.ground_truth_evaluated is True
