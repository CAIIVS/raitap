"""Tests for visualiser implementations"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pytest
import torch

from raitap.transparency.contracts import (
    ExplanationOutputSpace,
    ExplanationPayloadKind,
    ExplanationScope,
    ExplanationSemantics,
    InputSpec,
    MethodFamily,
    OutputSpaceSpec,
    ScopeDefinitionStep,
    VisualisationContext,
)
from raitap.transparency.visualisers import (
    BaseVisualiser,
    CaptumImageVisualiser,
    CaptumTextVisualiser,
    CaptumTimeSeriesVisualiser,
    ShapBarVisualiser,
    ShapBeeswarmVisualiser,
    ShapForceVisualiser,
    ShapImageVisualiser,
    ShapWaterfallVisualiser,
    TabularBarChartVisualiser,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from matplotlib.figure import Figure


def _explanation(
    *,
    scope: ExplanationScope = ExplanationScope.LOCAL,
    payload_kind: ExplanationPayloadKind = ExplanationPayloadKind.ATTRIBUTIONS,
    output_space: ExplanationOutputSpace = ExplanationOutputSpace.INPUT_FEATURES,
    method_families: frozenset[MethodFamily] = frozenset({MethodFamily.GRADIENT}),
    input_kind: str | None = "tabular",
    input_layout: str | None = "(B, F)",
    input_metadata: dict[str, object] | None = None,
    output_layout: str | None = "(B, F)",
    shape: tuple[int, ...] | None = (4, 10),
) -> SimpleNamespace:
    return SimpleNamespace(
        semantics=ExplanationSemantics(
            scope=scope,
            scope_definition_step=ScopeDefinitionStep.EXPLAINER_OUTPUT,
            payload_kind=payload_kind,
            method_families=method_families,
            target=None,
            sample_selection=None,
            input_spec=InputSpec(
                kind=input_kind,
                shape=shape,
                layout=input_layout,
                metadata=input_metadata,
            )
            if input_kind is not None or input_layout is not None or input_metadata is not None
            else None,
            output_space=OutputSpaceSpec(
                space=output_space,
                shape=shape,
                layout=output_layout,
            ),
        )
    )


class _ContractVisualiser(BaseVisualiser):
    supported_payload_kinds = frozenset({ExplanationPayloadKind.ATTRIBUTIONS})
    supported_scopes = frozenset({ExplanationScope.LOCAL})
    supported_output_spaces = frozenset({ExplanationOutputSpace.INPUT_FEATURES})
    supported_method_families = frozenset({MethodFamily.GRADIENT})

    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Figure:
        del attributions, inputs, kwargs
        fig, _ax = plt.subplots(figsize=(1, 1))
        return fig


class TestBaseVisualiserContract:
    def test_validate_explanation_accepts_matching_semantics(self) -> None:
        _ContractVisualiser().validate_explanation(
            _explanation(),
            torch.zeros(4, 10),
            None,
        )

    @pytest.mark.parametrize(
        ("explanation", "dimension"),
        [
            (
                _explanation(payload_kind=ExplanationPayloadKind.STRUCTURED),
                "payload kind",
            ),
            (
                _explanation(scope=ExplanationScope.COHORT),
                "scope",
            ),
            (
                _explanation(output_space=ExplanationOutputSpace.TOKEN_SEQUENCE),
                "output space",
            ),
            (
                _explanation(method_families=frozenset({MethodFamily.TREE})),
                "method family",
            ),
        ],
    )
    def test_validate_explanation_rejects_failed_dimension(
        self,
        explanation: SimpleNamespace,
        dimension: str,
    ) -> None:
        with pytest.raises(ValueError, match=rf"_ContractVisualiser.*{dimension}"):
            _ContractVisualiser().validate_explanation(explanation, torch.zeros(4, 10), None)


class TestCaptumImageVisualiser:
    """Test CaptumImageVisualiser"""

    def test_initialization(self) -> None:
        visualiser = CaptumImageVisualiser()
        assert visualiser is not None
        assert visualiser.include_original_image is True

    @pytest.mark.parametrize(
        "explanation",
        [
            _explanation(
                input_kind="image",
                input_layout="NCHW",
                output_layout="NCHW",
                shape=(2, 3, 32, 32),
                method_families=frozenset({MethodFamily.GRADIENT}),
            ),
            _explanation(
                input_kind="image",
                input_layout="NCHW",
                output_layout="NCHW",
                output_space=ExplanationOutputSpace.IMAGE_SPATIAL_MAP,
                shape=(2, 1, 8, 8),
                method_families=frozenset({MethodFamily.GRADIENT, MethodFamily.CAM}),
            ),
            _explanation(
                input_kind=None,
                input_layout="NCHW",
                input_metadata={"modality": "image"},
                output_layout="NCHW",
                shape=(2, 3, 32, 32),
                method_families=frozenset({MethodFamily.GRADIENT}),
            ),
        ],
    )
    def test_validate_explanation_accepts_image_and_cam_semantics(
        self,
        explanation: SimpleNamespace,
    ) -> None:
        CaptumImageVisualiser().validate_explanation(explanation, torch.zeros(2, 3, 32, 32), None)

    def test_validate_explanation_rejects_tabular_semantics(self) -> None:
        explanation = _explanation(input_kind="tabular", shape=(2, 10))

        with pytest.raises(ValueError, match=r"CaptumImageVisualiser.*input metadata"):
            CaptumImageVisualiser().validate_explanation(explanation, torch.zeros(2, 10), None)

    def test_validate_explanation_rejects_contradictory_non_image_nchw_metadata(self) -> None:
        explanation = _explanation(
            input_kind="tabular",
            input_layout="NCHW",
            output_layout="NCHW",
            shape=(2, 3, 32, 32),
        )

        with pytest.raises(ValueError, match=r"CaptumImageVisualiser.*input metadata"):
            CaptumImageVisualiser().validate_explanation(
                explanation,
                torch.zeros(2, 3, 32, 32),
                None,
            )

    def test_validate_explanation_rejects_missing_shape_metadata(self) -> None:
        explanation = _explanation(
            input_kind="image",
            input_layout="NCHW",
            output_layout="NCHW",
            shape=None,
        )

        with pytest.raises(ValueError, match=r"CaptumImageVisualiser.*input metadata"):
            CaptumImageVisualiser().validate_explanation(explanation, object(), None)  # type: ignore[arg-type]

    @pytest.mark.usefixtures("needs_captum")
    def test_visualise_tensor(self, sample_images: torch.Tensor) -> None:
        visualiser = CaptumImageVisualiser(method="heat_map")
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(attributions)
        assert fig is not None
        assert len(fig.axes) == 8  # 2 axes per sample (original + attribution) for 4 samples

    @pytest.mark.usefixtures("needs_captum")
    def test_max_samples_limit(self) -> None:
        visualiser = CaptumImageVisualiser(method="heat_map")
        large_batch = torch.randn(64, 3, 32, 32)

        fig = visualiser.visualise(large_batch, max_samples=4)
        assert len(fig.axes) == 8

    @pytest.mark.usefixtures("needs_captum")
    def test_overlay_with_inputs(self, sample_images: torch.Tensor) -> None:
        visualiser = CaptumImageVisualiser()
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(attributions, inputs=sample_images)
        assert fig is not None
        titles = [ax.get_title() for ax in fig.axes[:2]]
        assert titles == ["Original Image", "Blended Heat Map"]

    @pytest.mark.usefixtures("needs_captum")
    def test_original_and_attribution_panels_have_equal_size(
        self, sample_images: torch.Tensor
    ) -> None:
        visualiser = CaptumImageVisualiser(method="heat_map", show_colorbar=True)
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(attributions, inputs=sample_images, max_samples=1)

        original_pos = fig.axes[0].get_position()
        attr_pos = fig.axes[1].get_position()
        assert original_pos.width == pytest.approx(attr_pos.width, rel=1e-3)
        assert original_pos.height == pytest.approx(attr_pos.height, rel=1e-3)

    @pytest.mark.usefixtures("needs_captum")
    def test_can_disable_original_image_panel(self, sample_images: torch.Tensor) -> None:
        visualiser = CaptumImageVisualiser(method="heat_map", include_original_image=False)
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(attributions, inputs=sample_images, max_samples=2)

        assert len(fig.axes) == 2
        assert fig.axes[0].get_title() == ""

    @pytest.mark.usefixtures("needs_captum")
    def test_original_image_method_avoids_duplicate_panels(
        self, sample_images: torch.Tensor
    ) -> None:
        visualiser = CaptumImageVisualiser(method="original_image")
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(attributions, inputs=sample_images, max_samples=1)

        assert len(fig.axes) == 2
        assert fig.axes[0].get_title() == ""

    @pytest.mark.usefixtures("needs_captum")
    def test_masked_image_resizes_low_res_attributions(self) -> None:
        visualiser = CaptumImageVisualiser(method="masked_image", sign="absolute_value")
        attributions = torch.randn(1, 1, 12, 12)
        inputs = torch.rand(1, 3, 450, 600)

        fig = visualiser.visualise(attributions, inputs=inputs, max_samples=1)
        assert fig is not None

    @pytest.mark.usefixtures("needs_captum")
    def test_save(self, sample_images: torch.Tensor, tmp_path: Path) -> None:
        visualiser = CaptumImageVisualiser(method="heat_map")
        attributions = torch.randn_like(sample_images)
        output_path = tmp_path / "test_output.png"

        visualiser.save(attributions, output_path)
        assert output_path.exists()

    @pytest.mark.usefixtures("needs_captum")
    def test_show_sample_names_sets_axis_titles(self, sample_images: torch.Tensor) -> None:
        visualiser = CaptumImageVisualiser(method="heat_map")
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(
            attributions,
            inputs=sample_images,
            max_samples=1,
            context=VisualisationContext(
                algorithm="",
                sample_names=["ISIC_0001", "ISIC_0002", "ISIC_0003"],
                show_sample_names=True,
            ),
        )

        titles = [ax.get_title() for ax in fig.axes[:2]]
        assert titles == ["Original Image: ISIC_0001", "Heat Map: ISIC_0001"]

    @pytest.mark.usefixtures("needs_captum")
    def test_show_sample_names_sets_titles_for_each_sample_pair(
        self, sample_images: torch.Tensor
    ) -> None:
        visualiser = CaptumImageVisualiser(method="heat_map")
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(
            attributions,
            inputs=sample_images,
            max_samples=2,
            context=VisualisationContext(
                algorithm="",
                sample_names=["ISIC_0001", "ISIC_0002", "ISIC_0003"],
                show_sample_names=True,
            ),
        )

        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert titles == [
            "Original Image: ISIC_0001",
            "Heat Map: ISIC_0001",
            "Original Image: ISIC_0002",
            "Heat Map: ISIC_0002",
        ]

    @pytest.mark.usefixtures("needs_captum")
    def test_sample_names_are_hidden_by_default(self, sample_images: torch.Tensor) -> None:
        visualiser = CaptumImageVisualiser(method="heat_map")
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(
            attributions,
            max_samples=1,
            context=VisualisationContext(
                algorithm="",
                sample_names=["ISIC_9999"],
                show_sample_names=False,
            ),
        )

        assert fig.axes[0].get_title() == ""

    @pytest.mark.usefixtures("needs_captum")
    def test_show_sample_names_prefixes_existing_title(self, sample_images: torch.Tensor) -> None:
        visualiser = CaptumImageVisualiser(method="heat_map")
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(
            attributions,
            inputs=sample_images,
            max_samples=1,
            context=VisualisationContext(
                algorithm="",
                sample_names=["ISIC_1234"],
                show_sample_names=True,
            ),
            title="LayerGradCAM",
        )

        titles = [ax.get_title() for ax in fig.axes[:2]]
        assert titles == ["Original Image: ISIC_1234", "LayerGradCAM: ISIC_1234"]

    @pytest.mark.usefixtures("needs_captum")
    def test_explicit_empty_title_is_preserved_in_paired_layout(
        self, sample_images: torch.Tensor
    ) -> None:
        visualiser = CaptumImageVisualiser(method="heat_map")
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(
            attributions,
            inputs=sample_images,
            max_samples=1,
            title="",
        )

        titles = [ax.get_title() for ax in fig.axes[:2]]
        assert titles == ["Original Image", ""]


class TestTabularBarChartVisualiser:
    """Test tabular visualiser"""

    def test_initialization(self) -> None:
        """Test visualiser can be initialized"""
        visualiser = TabularBarChartVisualiser()
        assert visualiser is not None

    def test_contract_produces_cohort_visualiser_summary(self) -> None:
        assert TabularBarChartVisualiser.produces_scope is ExplanationScope.COHORT
        assert (
            TabularBarChartVisualiser.scope_definition_step
            is ScopeDefinitionStep.VISUALISER_SUMMARY
        )
        assert TabularBarChartVisualiser.visual_summary is not None
        assert TabularBarChartVisualiser.visual_summary.aggregation == "mean_absolute_attribution"

    def test_validate_explanation_accepts_tabular_bf_semantics(self) -> None:
        TabularBarChartVisualiser().validate_explanation(
            _explanation(input_kind="tabular", shape=(4, 10)),
            torch.zeros(4, 10),
            None,
        )

    @pytest.mark.parametrize(
        "explanation",
        [
            _explanation(
                input_kind="image",
                input_layout="NCHW",
                output_layout="NCHW",
                shape=(2, 3, 32, 32),
            ),
            _explanation(
                input_kind="time_series",
                input_layout="B,T,C",
                output_layout="B,T,C",
                shape=(2, 12, 3),
            ),
            _explanation(
                input_kind="text",
                input_layout="TOKENS",
                output_layout="TOKENS",
                output_space=ExplanationOutputSpace.TOKEN_SEQUENCE,
                shape=(12,),
            ),
            _explanation(input_kind=None, input_layout=None, output_layout=None, shape=(2, 10)),
        ],
    )
    def test_validate_explanation_rejects_non_tabular_layouts(
        self,
        explanation: SimpleNamespace,
    ) -> None:
        with pytest.raises(ValueError, match="TabularBarChartVisualiser"):
            TabularBarChartVisualiser().validate_explanation(explanation, torch.zeros(2, 10), None)

    def test_initialization_with_feature_names(self, feature_names: list[str]) -> None:
        """Test initialization with feature names"""
        visualiser = TabularBarChartVisualiser(feature_names=feature_names)
        assert visualiser.feature_names == feature_names

    def test_visualize_tensor(self, sample_tabular: torch.Tensor) -> None:
        """Test visualization with torch.Tensor"""
        visualiser = TabularBarChartVisualiser()

        fig = visualiser.visualise(sample_tabular)
        assert fig is not None

    def test_feature_names_display(
        self, sample_tabular: torch.Tensor, feature_names: list[str]
    ) -> None:
        """Test feature names are displayed"""
        visualiser = TabularBarChartVisualiser(feature_names=feature_names)

        fig = visualiser.visualise(sample_tabular)
        assert fig is not None

    def test_save(self, sample_tabular: torch.Tensor, tmp_path: Path) -> None:
        """Test save functionality"""
        visualiser = TabularBarChartVisualiser()
        output_path = tmp_path / "test_tabular.png"

        visualiser.save(sample_tabular, output_path)
        assert output_path.exists()


class TestCaptumTimeSeriesVisualiser:
    """Test CaptumTimeSeriesVisualiser"""

    def test_initialization(self) -> None:
        visualiser = CaptumTimeSeriesVisualiser()
        assert visualiser is not None

    def test_validate_explanation_accepts_explicit_time_series_metadata(self) -> None:
        explanation = _explanation(
            input_kind="time_series",
            input_layout="B,T,C",
            output_layout="B,T,C",
            shape=(2, 12, 3),
        )

        CaptumTimeSeriesVisualiser().validate_explanation(explanation, torch.zeros(2, 12, 3), None)

    @pytest.mark.parametrize(
        "explanation",
        [
            _explanation(input_kind="time_series", input_layout="TOKENS", output_layout="TOKENS"),
            _explanation(input_kind="time_series", input_layout="B,F", output_layout="B,F"),
            _explanation(
                input_kind="time_series",
                input_layout="B,T,C",
                output_layout="B,T,C",
                shape=(12,),
            ),
        ],
    )
    def test_validate_explanation_rejects_incompatible_layout_or_shape(
        self,
        explanation: SimpleNamespace,
    ) -> None:
        with pytest.raises(ValueError, match="CaptumTimeSeriesVisualiser"):
            CaptumTimeSeriesVisualiser().validate_explanation(
                explanation,
                torch.zeros(12),
                None,
            )

    @pytest.mark.parametrize(
        "method_families",
        [
            frozenset({MethodFamily.TREE}),
            frozenset({MethodFamily.CAM}),
        ],
    )
    def test_validate_explanation_rejects_unsupported_method_families(
        self,
        method_families: frozenset[MethodFamily],
    ) -> None:
        explanation = _explanation(
            input_kind="time_series",
            input_layout="B,T,C",
            output_layout="B,T,C",
            shape=(2, 12, 3),
            method_families=method_families,
        )

        with pytest.raises(ValueError, match=r"CaptumTimeSeriesVisualiser.*method family"):
            CaptumTimeSeriesVisualiser().validate_explanation(
                explanation,
                torch.zeros(2, 12, 3),
                None,
            )

    @pytest.mark.parametrize(
        "explanation",
        [
            _explanation(input_kind=None, input_layout=None),
            _explanation(input_kind="tabular", shape=(2, 10)),
            _explanation(
                input_kind="text",
                input_layout="TOKENS",
                output_layout="TOKENS",
                output_space=ExplanationOutputSpace.TOKEN_SEQUENCE,
                shape=(12,),
            ),
        ],
    )
    def test_validate_explanation_rejects_missing_or_incompatible_metadata(
        self,
        explanation: SimpleNamespace,
    ) -> None:
        with pytest.raises(ValueError, match="CaptumTimeSeriesVisualiser"):
            CaptumTimeSeriesVisualiser().validate_explanation(
                explanation,
                torch.zeros(2, 12, 3),
                None,
            )

    @pytest.mark.usefixtures("needs_captum")
    def test_visualise_requires_inputs(self, sample_timeseries: torch.Tensor) -> None:
        """visualise() requires inputs alongside attributions."""
        visualiser = CaptumTimeSeriesVisualiser()
        attributions = torch.randn_like(sample_timeseries)
        with pytest.raises(ValueError, match="requires `inputs`"):
            visualiser.visualise(attributions)

    @pytest.mark.usefixtures("needs_captum")
    def test_visualise_with_inputs(self, sample_timeseries: torch.Tensor) -> None:
        """Returns a Figure when inputs are supplied."""
        visualiser = CaptumTimeSeriesVisualiser()
        attributions = torch.randn_like(sample_timeseries)

        fig = visualiser.visualise(attributions, inputs=sample_timeseries)
        assert fig is not None

    @pytest.mark.usefixtures("needs_captum")
    def test_save(self, sample_timeseries: torch.Tensor, tmp_path: Path) -> None:
        visualiser = CaptumTimeSeriesVisualiser()
        attributions = torch.randn_like(sample_timeseries)
        output_path = tmp_path / "timeseries.png"

        visualiser.save(attributions, output_path, inputs=sample_timeseries)
        assert output_path.exists()


class TestCaptumTextVisualiser:
    """Test CaptumTextVisualiser (pure matplotlib — no captum dependency)."""

    def test_initialization(self) -> None:
        visualiser = CaptumTextVisualiser()
        assert visualiser is not None

    def test_validate_explanation_accepts_token_sequence_metadata(self) -> None:
        explanation = _explanation(
            input_kind="text",
            input_layout="TOKENS",
            output_layout="TOKENS",
            output_space=ExplanationOutputSpace.TOKEN_SEQUENCE,
            shape=(12,),
        )

        CaptumTextVisualiser().validate_explanation(explanation, torch.zeros(12), None)

    @pytest.mark.parametrize(
        "explanation",
        [
            _explanation(
                input_kind="text",
                input_layout="TOKENS",
                output_layout="TOKENS",
                output_space=ExplanationOutputSpace.TOKEN_SEQUENCE,
                shape=(2, 12),
            ),
            _explanation(
                input_kind="text",
                input_layout="B,F",
                output_layout="B,F",
                output_space=ExplanationOutputSpace.TOKEN_SEQUENCE,
                shape=(2, 10),
            ),
        ],
    )
    def test_validate_explanation_rejects_incompatible_text_layout_or_shape(
        self,
        explanation: SimpleNamespace,
    ) -> None:
        with pytest.raises(ValueError, match="CaptumTextVisualiser"):
            CaptumTextVisualiser().validate_explanation(
                explanation,
                torch.zeros(2, 10),
                None,
            )

    @pytest.mark.parametrize(
        "method_families",
        [
            frozenset({MethodFamily.TREE}),
            frozenset({MethodFamily.CAM}),
        ],
    )
    def test_validate_explanation_rejects_unsupported_method_families(
        self,
        method_families: frozenset[MethodFamily],
    ) -> None:
        explanation = _explanation(
            input_kind="text",
            input_layout="TOKENS",
            output_layout="TOKENS",
            output_space=ExplanationOutputSpace.TOKEN_SEQUENCE,
            shape=(12,),
            method_families=method_families,
        )

        with pytest.raises(ValueError, match=r"CaptumTextVisualiser.*method family"):
            CaptumTextVisualiser().validate_explanation(explanation, torch.zeros(12), None)

    @pytest.mark.parametrize(
        "explanation",
        [
            _explanation(input_kind=None, input_layout=None),
            _explanation(input_kind="tabular", shape=(2, 10)),
            _explanation(
                input_kind="time_series",
                input_layout="B,T,C",
                output_layout="B,T,C",
                shape=(2, 12, 3),
            ),
            _explanation(
                input_kind="time_series",
                input_layout="TOKENS",
                output_layout="TOKENS",
                output_space=ExplanationOutputSpace.TOKEN_SEQUENCE,
                shape=(12,),
            ),
        ],
    )
    def test_validate_explanation_rejects_missing_or_incompatible_metadata(
        self,
        explanation: SimpleNamespace,
    ) -> None:
        with pytest.raises(ValueError, match="CaptumTextVisualiser"):
            CaptumTextVisualiser().validate_explanation(explanation, torch.zeros(12), None)

    def test_visualise_1d_tensor(self, sample_text_attributions: torch.Tensor) -> None:
        """1-D attribution tensor produces a figure."""
        visualiser = CaptumTextVisualiser()
        fig = visualiser.visualise(sample_text_attributions)
        assert fig is not None

    def test_visualise_with_token_labels(
        self, sample_text_attributions: torch.Tensor, token_labels: list[str]
    ) -> None:
        """Figure is produced with token labels supplied."""
        visualiser = CaptumTextVisualiser()
        fig = visualiser.visualise(sample_text_attributions, token_labels=token_labels)
        assert fig is not None

    def test_save(self, sample_text_attributions: torch.Tensor, tmp_path: Path) -> None:
        visualiser = CaptumTextVisualiser()
        output_path = tmp_path / "text.png"
        visualiser.save(sample_text_attributions, output_path)
        assert output_path.exists()


class TestShapBarVisualiser:
    """Test ShapBarVisualiser"""

    def test_initialization(self) -> None:
        visualiser = ShapBarVisualiser()
        assert visualiser is not None

    def test_contract_produces_cohort_visualiser_summary(self) -> None:
        assert ShapBarVisualiser.produces_scope is ExplanationScope.COHORT
        assert ShapBarVisualiser.scope_definition_step is ScopeDefinitionStep.VISUALISER_SUMMARY
        assert ShapBarVisualiser.visual_summary is not None
        assert ShapBarVisualiser.visual_summary.aggregation == "mean_absolute_attribution"

    def test_validate_explanation_accepts_tabular_shap_semantics(self) -> None:
        explanation = _explanation(
            method_families=frozenset({MethodFamily.SHAPLEY, MethodFamily.TREE}),
        )

        ShapBarVisualiser().validate_explanation(explanation, torch.zeros(4, 10), None)

    def test_validate_explanation_rejects_image_shap_values(self) -> None:
        explanation = _explanation(
            input_kind="image",
            input_layout="NCHW",
            output_layout="NCHW",
            shape=(2, 3, 32, 32),
            method_families=frozenset({MethodFamily.SHAPLEY, MethodFamily.GRADIENT}),
        )

        with pytest.raises(ValueError, match=r"ShapBarVisualiser.*tabular layout"):
            ShapBarVisualiser().validate_explanation(explanation, torch.zeros(2, 3, 32, 32), None)

    def test_validate_explanation_rejects_shape_only_semantics(self) -> None:
        explanation = _explanation(
            input_kind=None,
            input_layout=None,
            output_layout=None,
            shape=(4, 10),
            method_families=frozenset({MethodFamily.SHAPLEY}),
        )

        with pytest.raises(ValueError, match=r"ShapBarVisualiser.*tabular layout"):
            ShapBarVisualiser().validate_explanation(explanation, torch.zeros(4, 10), None)

    @pytest.mark.usefixtures("needs_shap")
    def test_visualise_tensor(self, sample_tabular: torch.Tensor) -> None:
        visualiser = ShapBarVisualiser()
        fig = visualiser.visualise(sample_tabular)
        assert fig is not None

    @pytest.mark.usefixtures("needs_shap")
    def test_visualise_with_feature_names(
        self,
        sample_tabular: torch.Tensor,
        feature_names: list[str],
    ) -> None:
        visualiser = ShapBarVisualiser(feature_names=feature_names)
        fig = visualiser.visualise(sample_tabular)
        assert fig is not None

    @pytest.mark.usefixtures("needs_shap")
    def test_save(self, sample_tabular: torch.Tensor, tmp_path: Path) -> None:
        visualiser = ShapBarVisualiser()
        output_path = tmp_path / "shap_bar.png"
        visualiser.save(sample_tabular, output_path)
        assert output_path.exists()


class TestShapBeeswarmVisualiser:
    """Test ShapBeeswarmVisualiser"""

    def test_initialization(self) -> None:
        visualiser = ShapBeeswarmVisualiser()
        assert visualiser is not None

    def test_contract_produces_cohort_distribution_summary(self) -> None:
        assert ShapBeeswarmVisualiser.produces_scope is ExplanationScope.COHORT
        assert (
            ShapBeeswarmVisualiser.scope_definition_step is ScopeDefinitionStep.VISUALISER_SUMMARY
        )
        assert ShapBeeswarmVisualiser.visual_summary is not None
        assert ShapBeeswarmVisualiser.visual_summary.aggregation == "distribution_summary"

    def test_validate_explanation_rejects_image_shap_values(self) -> None:
        explanation = _explanation(
            input_kind="image",
            input_layout="NCHW",
            output_layout="NCHW",
            shape=(2, 3, 32, 32),
            method_families=frozenset({MethodFamily.SHAPLEY, MethodFamily.GRADIENT}),
        )

        with pytest.raises(ValueError, match=r"ShapBeeswarmVisualiser.*tabular layout"):
            ShapBeeswarmVisualiser().validate_explanation(
                explanation,
                torch.zeros(2, 3, 32, 32),
                None,
            )

    def test_validate_explanation_rejects_shape_only_semantics(self) -> None:
        explanation = _explanation(
            input_kind=None,
            input_layout=None,
            output_layout=None,
            shape=(4, 10),
            method_families=frozenset({MethodFamily.SHAPLEY}),
        )

        with pytest.raises(ValueError, match=r"ShapBeeswarmVisualiser.*tabular layout"):
            ShapBeeswarmVisualiser().validate_explanation(
                explanation,
                torch.zeros(4, 10),
                None,
            )

    @pytest.mark.usefixtures("needs_shap")
    def test_visualise_tensor(self, sample_tabular: torch.Tensor) -> None:
        visualiser = ShapBeeswarmVisualiser()
        fig = visualiser.visualise(sample_tabular)
        assert fig is not None

    @pytest.mark.usefixtures("needs_shap")
    def test_save(self, sample_tabular: torch.Tensor, tmp_path: Path) -> None:
        visualiser = ShapBeeswarmVisualiser()
        output_path = tmp_path / "shap_beeswarm.png"
        visualiser.save(sample_tabular, output_path)
        assert output_path.exists()


class TestShapWaterfallVisualiser:
    """Test ShapWaterfallVisualiser"""

    def test_initialization(self) -> None:
        visualiser = ShapWaterfallVisualiser()
        assert visualiser is not None

    @pytest.mark.usefixtures("needs_shap")
    def test_visualise_single_sample(self, sample_tabular: torch.Tensor) -> None:
        """Waterfall chart for sample_index=0 of the batch."""
        visualiser = ShapWaterfallVisualiser()
        fig = visualiser.visualise(sample_tabular)
        assert fig is not None

    @pytest.mark.usefixtures("needs_shap")
    def test_visualise_with_feature_names(
        self,
        sample_tabular: torch.Tensor,
        feature_names: list[str],
    ) -> None:
        """Test visualization with feature names"""
        visualiser = ShapWaterfallVisualiser(feature_names=feature_names)
        fig = visualiser.visualise(sample_tabular)
        assert fig is not None

    @pytest.mark.usefixtures("needs_shap")
    def test_save(self, sample_tabular: torch.Tensor, tmp_path: Path) -> None:
        visualiser = ShapWaterfallVisualiser()
        output_path = tmp_path / "shap_waterfall.png"
        visualiser.save(sample_tabular, output_path)
        assert output_path.exists()


class TestShapForceVisualiser:
    """Test ShapForceVisualiser"""

    def test_initialization(self) -> None:
        visualiser = ShapForceVisualiser()
        assert visualiser is not None

    @pytest.mark.usefixtures("needs_shap")
    def test_visualise_single_sample(self, sample_tabular: torch.Tensor) -> None:
        """Force plot for sample_index=0 of the batch."""
        visualiser = ShapForceVisualiser()
        fig = visualiser.visualise(sample_tabular)
        assert fig is not None

    @pytest.mark.usefixtures("needs_shap")
    def test_save(self, sample_tabular: torch.Tensor, tmp_path: Path) -> None:
        visualiser = ShapForceVisualiser()
        output_path = tmp_path / "shap_force.png"
        visualiser.save(sample_tabular, output_path)
        assert output_path.exists()


class TestShapImageVisualiser:
    """Test ShapImageVisualiser (restricted to GradientExplainer / DeepExplainer)."""

    def test_initialization(self) -> None:
        visualiser = ShapImageVisualiser()
        assert visualiser is not None

    def test_compatible_algorithms_restriction(self) -> None:
        assert ShapImageVisualiser.compatible_algorithms == frozenset(
            {"GradientExplainer", "DeepExplainer"}
        )

    def test_validate_explanation_accepts_gradient_image_shap_semantics(self) -> None:
        explanation = _explanation(
            input_kind="image",
            input_layout="NCHW",
            output_layout="NCHW",
            shape=(2, 3, 32, 32),
            method_families=frozenset({MethodFamily.SHAPLEY, MethodFamily.GRADIENT}),
        )

        ShapImageVisualiser().validate_explanation(explanation, torch.zeros(2, 3, 32, 32), None)

    def test_validate_explanation_accepts_explicit_image_metadata_without_kind(self) -> None:
        explanation = _explanation(
            input_kind=None,
            input_layout="NCHW",
            input_metadata={"modality": "image"},
            output_layout="NCHW",
            shape=(2, 3, 32, 32),
            method_families=frozenset({MethodFamily.SHAPLEY, MethodFamily.GRADIENT}),
        )

        ShapImageVisualiser().validate_explanation(explanation, torch.zeros(2, 3, 32, 32), None)

    def test_validate_explanation_rejects_contradictory_non_image_nchw_metadata(self) -> None:
        explanation = _explanation(
            input_kind="tabular",
            input_layout="NCHW",
            output_layout="NCHW",
            shape=(2, 3, 32, 32),
            method_families=frozenset({MethodFamily.SHAPLEY, MethodFamily.GRADIENT}),
        )

        with pytest.raises(ValueError, match=r"ShapImageVisualiser.*input metadata"):
            ShapImageVisualiser().validate_explanation(
                explanation,
                torch.zeros(2, 3, 32, 32),
                None,
            )

    def test_validate_explanation_rejects_missing_shape_metadata(self) -> None:
        explanation = _explanation(
            input_kind="image",
            input_layout="NCHW",
            output_layout="NCHW",
            shape=None,
            method_families=frozenset({MethodFamily.SHAPLEY, MethodFamily.GRADIENT}),
        )

        with pytest.raises(ValueError, match=r"ShapImageVisualiser.*input metadata"):
            ShapImageVisualiser().validate_explanation(explanation, object(), None)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "method_families",
        [
            frozenset({MethodFamily.SHAPLEY, MethodFamily.PERTURBATION}),
            frozenset({MethodFamily.SHAPLEY, MethodFamily.TREE}),
        ],
    )
    def test_validate_explanation_rejects_non_gradient_shap_semantics(
        self,
        method_families: frozenset[MethodFamily],
    ) -> None:
        explanation = _explanation(
            input_kind="image",
            input_layout="NCHW",
            output_layout="NCHW",
            shape=(2, 3, 32, 32),
            method_families=method_families,
        )

        with pytest.raises(ValueError, match=r"ShapImageVisualiser.*method family"):
            ShapImageVisualiser().validate_explanation(
                explanation,
                torch.zeros(2, 3, 32, 32),
                None,
            )

    @pytest.mark.usefixtures("needs_shap")
    def test_visualise_image_batch(self, sample_images: torch.Tensor) -> None:
        """Accepts a (B, C, H, W) attribution tensor."""
        visualiser = ShapImageVisualiser()
        attributions = torch.randn_like(sample_images)
        fig = visualiser.visualise(attributions, inputs=sample_images)
        assert fig is not None
        titles = [ax.get_title() for ax in fig.axes[:2]]
        assert titles == ["Original Image", "SHAP Image"]
        plt.close(fig)

    @pytest.mark.usefixtures("needs_shap")
    def test_show_sample_names_sets_axis_titles(self, sample_images: torch.Tensor) -> None:
        visualiser = ShapImageVisualiser()
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(
            attributions,
            inputs=sample_images,
            max_samples=1,
            context=VisualisationContext(
                algorithm="GradientExplainer",
                sample_names=["ISIC_0001", "ISIC_0002"],
                show_sample_names=True,
            ),
        )

        titles = [ax.get_title() for ax in fig.axes[:2]]
        assert titles == [
            "Original Image: ISIC_0001",
            "GradientExplainer (SHAP): ISIC_0001",
        ]
        plt.close(fig)

    @pytest.mark.usefixtures("needs_shap")
    def test_show_sample_names_sets_titles_for_each_sample_pair(
        self, sample_images: torch.Tensor
    ) -> None:
        visualiser = ShapImageVisualiser(show_colorbar=False)
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(
            attributions,
            inputs=sample_images,
            max_samples=2,
            context=VisualisationContext(
                algorithm="DeepExplainer",
                sample_names=["ISIC_0001", "ISIC_0002", "ISIC_0003"],
                show_sample_names=True,
            ),
        )

        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert titles == [
            "Original Image: ISIC_0001",
            "DeepExplainer (SHAP): ISIC_0001",
            "Original Image: ISIC_0002",
            "DeepExplainer (SHAP): ISIC_0002",
        ]
        plt.close(fig)

    @pytest.mark.usefixtures("needs_shap")
    def test_show_sample_names_prefixes_existing_title(self, sample_images: torch.Tensor) -> None:
        visualiser = ShapImageVisualiser()
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(
            attributions,
            inputs=sample_images,
            max_samples=1,
            context=VisualisationContext(
                algorithm="GradientExplainer",
                sample_names=["ISIC_1234"],
                show_sample_names=True,
            ),
            title="Custom SHAP",
        )

        titles = [ax.get_title() for ax in fig.axes[:2]]
        assert titles == ["Original Image: ISIC_1234", "Custom SHAP: ISIC_1234"]
        plt.close(fig)

    @pytest.mark.usefixtures("needs_shap")
    def test_explicit_empty_title_is_preserved_in_paired_layout(
        self, sample_images: torch.Tensor
    ) -> None:
        visualiser = ShapImageVisualiser()
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(
            attributions,
            inputs=sample_images,
            max_samples=1,
            context=VisualisationContext(
                algorithm="GradientExplainer",
                sample_names=None,
                show_sample_names=False,
            ),
            title="",
        )

        titles = [ax.get_title() for ax in fig.axes[:2]]
        assert titles == ["Original Image", ""]
        plt.close(fig)

    @pytest.mark.usefixtures("needs_shap")
    def test_explicit_empty_title_is_preserved_when_showing_sample_names(
        self, sample_images: torch.Tensor
    ) -> None:
        visualiser = ShapImageVisualiser()
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(
            attributions,
            inputs=sample_images,
            max_samples=1,
            context=VisualisationContext(
                algorithm="GradientExplainer",
                sample_names=["ISIC_0001"],
                show_sample_names=True,
            ),
            title="",
        )

        titles = [ax.get_title() for ax in fig.axes[:2]]
        assert titles == ["Original Image: ISIC_0001", ""]
        plt.close(fig)

    @pytest.mark.usefixtures("needs_shap")
    def test_save(self, sample_images: torch.Tensor, tmp_path: Path) -> None:
        visualiser = ShapImageVisualiser()
        attributions = torch.randn_like(sample_images)
        output_path = tmp_path / "shap_image.png"
        visualiser.save(attributions, output_path, inputs=sample_images)
        assert output_path.exists()
