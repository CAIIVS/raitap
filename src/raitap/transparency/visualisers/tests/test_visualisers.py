"""Tests for visualiser implementations"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
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
    StructuredPayload,
    StructuredPayloadKind,
    VisualisationContext,
)
from raitap.transparency.visualisers import (
    BaseVisualiser,
    CaptumImageVisualiser,
    CaptumTextVisualiser,
    CaptumTimeSeriesVisualiser,
    InputThumbnailVisualiser,
    ShapBarVisualiser,
    ShapBeeswarmVisualiser,
    ShapForceVisualiser,
    ShapImageVisualiser,
    ShapWaterfallVisualiser,
    TabularBarChartVisualiser,
)

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet
    from pathlib import Path
    from typing import Any

    from matplotlib.figure import Figure


def _explanation(
    *,
    scope: ExplanationScope = ExplanationScope.LOCAL,
    payload_kind: ExplanationPayloadKind = ExplanationPayloadKind.ATTRIBUTIONS,
    output_space: ExplanationOutputSpace = ExplanationOutputSpace.INPUT_FEATURES,
    method_families: AbstractSet[MethodFamily] = frozenset({MethodFamily.GRADIENT}),
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

    def test_original_input_contract_defaults_to_no_embedded_original(self) -> None:
        visualiser = _ContractVisualiser()

        assert type(visualiser).embeds_original_input is False
        assert visualiser.renders_attribution_only_when_original_hidden() is True

    @pytest.mark.parametrize(
        ("explanation", "dimension"),
        [
            (
                _explanation(payload_kind=ExplanationPayloadKind.STRUCTURED),
                "payload kind",
            ),
            (
                _explanation(scope=ExplanationScope.AGGREGATED),
                "scope",
            ),
            (
                _explanation(output_space=ExplanationOutputSpace.TOKEN_SEQUENCE),
                "output space",
            ),
            (
                _explanation(method_families={MethodFamily.TREE}),
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
                method_families={MethodFamily.GRADIENT},
            ),
            _explanation(
                input_kind="image",
                input_layout="NCHW",
                output_layout="NCHW",
                output_space=ExplanationOutputSpace.IMAGE_SPATIAL_MAP,
                shape=(2, 1, 8, 8),
                method_families={MethodFamily.GRADIENT, MethodFamily.CAM},
            ),
            _explanation(
                input_kind=None,
                input_layout="NCHW",
                input_metadata={"modality": "image"},
                output_layout="NCHW",
                shape=(2, 3, 32, 32),
                method_families={MethodFamily.GRADIENT},
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
        assert len(fig.axes) >= 4  # at least one axes per sample

    @pytest.mark.usefixtures("needs_captum")
    def test_all_zero_attribution_renders_flat_panel_with_note(self) -> None:
        # An all-zero map is a valid explainer output ("no positive evidence for
        # the predicted class"), not malformed input. Captum's normaliser asserts
        # ``scale factor = 0`` on it; the visualiser must instead render the map
        # honestly as a flat panel annotated with a sign-aware note.
        visualiser = CaptumImageVisualiser(method="heat_map", sign="positive")
        attributions = torch.zeros(1, 3, 32, 32)

        fig = visualiser.visualise(attributions)

        assert fig is not None
        titles = " ".join(ax.get_title() for ax in fig.axes)
        assert "no positive attribution" in titles

    @pytest.mark.usefixtures("needs_captum")
    def test_zero_attribution_sample_renders_alongside_valid_one(
        self, sample_images: torch.Tensor
    ) -> None:
        # One degenerate sample must not abort the valid samples in the batch.
        visualiser = CaptumImageVisualiser(method="heat_map", sign="positive")
        attributions = torch.randn(2, 3, 32, 32).abs()
        attributions[0] = 0.0  # first sample is the degenerate all-zero map

        fig = visualiser.visualise(attributions, inputs=sample_images[:2], max_samples=2)

        assert fig is not None
        assert len(fig.axes) >= 2

    @pytest.mark.usefixtures("needs_captum")
    def test_max_samples_limit(self) -> None:
        visualiser = CaptumImageVisualiser(method="heat_map")
        large_batch = torch.randn(64, 3, 32, 32)

        fig = visualiser.visualise(large_batch, max_samples=4)
        assert len(fig.axes) >= 4

    @pytest.mark.usefixtures("needs_captum")
    def test_overlay_with_inputs(self, sample_images: torch.Tensor) -> None:
        visualiser = CaptumImageVisualiser()
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(attributions, inputs=sample_images)
        assert fig is not None
        titles = [ax.get_title() for ax in fig.axes[:2]]
        assert titles == ["Original Image", "Blended Heat Map"]

    @pytest.mark.usefixtures("needs_captum")
    def test_colorbar_is_labelled_single_panel(self, sample_images: torch.Tensor) -> None:
        # include_original_image=False -> single attribution axis with captum's
        # native colorbar; it must carry the normalised-attribution label so the
        # bar's values are not unexplained.
        visualiser = CaptumImageVisualiser(
            method="heat_map", show_colorbar=True, include_original_image=False
        )
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(attributions, inputs=sample_images, max_samples=1)

        labels = [ax.get_xlabel() for ax in fig.axes] + [ax.get_ylabel() for ax in fig.axes]
        assert "Normalised attribution" in labels

    @pytest.mark.usefixtures("needs_captum")
    def test_colorbar_is_labelled_paired_panel(self, sample_images: torch.Tensor) -> None:
        # include_original_image=True -> dedicated colorbar axis (cax); same label.
        visualiser = CaptumImageVisualiser(
            method="heat_map", show_colorbar=True, include_original_image=True
        )
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(attributions, inputs=sample_images, max_samples=1)

        labels = [ax.get_xlabel() for ax in fig.axes] + [ax.get_ylabel() for ax in fig.axes]
        assert "Normalised attribution" in labels

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

        assert len(fig.axes) >= 2
        assert fig.axes[0].get_title() == ""

    @pytest.mark.usefixtures("needs_captum")
    def test_can_disable_original_image_panel_with_neutral_runtime_kwarg(
        self,
        sample_images: torch.Tensor,
    ) -> None:
        visualiser = CaptumImageVisualiser(method="heat_map")
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(
            attributions,
            inputs=sample_images,
            max_samples=1,
            include_original_input=False,
        )

        assert "Original Image" not in [ax.get_title() for ax in fig.axes]

    @pytest.mark.usefixtures("needs_captum")
    def test_runtime_include_original_image_alias_warns(
        self,
        sample_images: torch.Tensor,
    ) -> None:
        visualiser = CaptumImageVisualiser(method="heat_map")
        attributions = torch.randn_like(sample_images)

        with pytest.warns(DeprecationWarning, match="include_original_input"):
            fig = visualiser.visualise(
                attributions,
                inputs=sample_images,
                max_samples=1,
                include_original_image=False,
            )

        assert "Original Image" not in [ax.get_title() for ax in fig.axes]

    @pytest.mark.usefixtures("needs_captum")
    def test_neutral_runtime_kwarg_wins_over_deprecated_alias_without_warning(
        self,
        sample_images: torch.Tensor,
    ) -> None:
        visualiser = CaptumImageVisualiser(method="heat_map")
        attributions = torch.randn_like(sample_images)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig = visualiser.visualise(
                attributions,
                inputs=sample_images,
                max_samples=1,
                include_original_input=False,
                include_original_image=True,
            )

        assert caught == []
        assert "Original Image" not in [ax.get_title() for ax in fig.axes]

    def test_masked_image_keeps_original_when_original_hidden_would_remove_attribution(
        self,
    ) -> None:
        visualiser = CaptumImageVisualiser(method="masked_image")

        assert visualiser.renders_attribution_only_when_original_hidden() is False

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
    @pytest.mark.parametrize("method", ["blended_heat_map", "heat_map"])
    def test_layer_methods_resize_low_res_attributions(self, method: str) -> None:
        visualiser = CaptumImageVisualiser(method=method, sign="positive")
        attributions = torch.randn(1, 1, 7, 7)
        inputs = torch.rand(1, 3, 224, 224)

        fig = visualiser.visualise(attributions, inputs=inputs, max_samples=1)

        attr_axes = [ax for ax in fig.axes if getattr(ax, "images", None)]
        heat_array = attr_axes[-1].images[-1].get_array()
        assert heat_array is not None
        assert heat_array.shape[:2] == (224, 224)

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

    def test_contract_produces_aggregated_visualiser_summary(self) -> None:
        assert TabularBarChartVisualiser.produces_scope is ExplanationScope.AGGREGATED
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
            {MethodFamily.TREE},
            {MethodFamily.CAM},
        ],
    )
    def test_validate_explanation_rejects_unsupported_method_families(
        self,
        method_families: AbstractSet[MethodFamily],
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

    def test_validate_explanation_accepts_batched_token_sequence(self) -> None:
        # A 2-D ``(B, T)`` batch of per-token scores is a valid token-sequence
        # layout (one row per sample), rendered as one bar panel per sample. (#340)
        explanation = _explanation(
            input_kind="text",
            input_layout="TOKENS",
            output_layout="TOKENS",
            output_space=ExplanationOutputSpace.TOKEN_SEQUENCE,
            shape=(2, 12),
        )
        CaptumTextVisualiser().validate_explanation(explanation, torch.zeros(2, 12), None)

    def test_visualise_rejects_token_labels_for_batched_attribution(self) -> None:
        # A single ``token_labels`` list reused across a (B, T) batch would
        # mislabel every sample after the first; reject it loudly (#99).
        with pytest.raises(ValueError, match="batched"):
            CaptumTextVisualiser().visualise(
                torch.zeros(3, 5), token_labels=[f"w{i}" for i in range(5)]
            )

    def test_visualise_batched_without_labels_renders(self) -> None:
        fig = CaptumTextVisualiser().visualise(torch.zeros(3, 5))
        assert fig is not None

    def test_visualise_rejects_3d_attribution(self) -> None:
        # An un-reduced (B, T, H) tensor must not be silently flattened.
        with pytest.raises(ValueError, match=r"1-D .* or 2-D"):
            CaptumTextVisualiser().visualise(torch.zeros(2, 4, 8))

    @pytest.mark.parametrize(
        "method_families",
        [
            {MethodFamily.TREE},
            {MethodFamily.CAM},
        ],
    )
    def test_validate_explanation_rejects_unsupported_method_families(
        self,
        method_families: AbstractSet[MethodFamily],
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

    def test_contract_produces_aggregated_visualiser_summary(self) -> None:
        assert ShapBarVisualiser.produces_scope is ExplanationScope.AGGREGATED
        assert ShapBarVisualiser.scope_definition_step is ScopeDefinitionStep.VISUALISER_SUMMARY
        assert ShapBarVisualiser.visual_summary is not None
        assert ShapBarVisualiser.visual_summary.aggregation == "mean_absolute_attribution"

    def test_validate_explanation_accepts_tabular_shap_semantics(self) -> None:
        explanation = _explanation(
            method_families={MethodFamily.SHAPLEY, MethodFamily.TREE},
        )

        ShapBarVisualiser().validate_explanation(explanation, torch.zeros(4, 10), None)

    def test_validate_explanation_rejects_image_shap_values(self) -> None:
        explanation = _explanation(
            input_kind="image",
            input_layout="NCHW",
            output_layout="NCHW",
            shape=(2, 3, 32, 32),
            method_families={MethodFamily.SHAPLEY, MethodFamily.GRADIENT},
        )

        with pytest.raises(ValueError, match=r"ShapBarVisualiser.*tabular layout"):
            ShapBarVisualiser().validate_explanation(explanation, torch.zeros(2, 3, 32, 32), None)

    def test_validate_explanation_rejects_shape_only_semantics(self) -> None:
        explanation = _explanation(
            input_kind=None,
            input_layout=None,
            output_layout=None,
            shape=(4, 10),
            method_families={MethodFamily.SHAPLEY},
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

    def test_contract_produces_aggregated_distribution_summary(self) -> None:
        assert ShapBeeswarmVisualiser.produces_scope is ExplanationScope.AGGREGATED
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
            method_families={MethodFamily.SHAPLEY, MethodFamily.GRADIENT},
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
            method_families={MethodFamily.SHAPLEY},
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
            method_families={MethodFamily.SHAPLEY, MethodFamily.GRADIENT},
        )

        ShapImageVisualiser().validate_explanation(explanation, torch.zeros(2, 3, 32, 32), None)

    def test_validate_explanation_accepts_explicit_image_metadata_without_kind(self) -> None:
        explanation = _explanation(
            input_kind=None,
            input_layout="NCHW",
            input_metadata={"modality": "image"},
            output_layout="NCHW",
            shape=(2, 3, 32, 32),
            method_families={MethodFamily.SHAPLEY, MethodFamily.GRADIENT},
        )

        ShapImageVisualiser().validate_explanation(explanation, torch.zeros(2, 3, 32, 32), None)

    def test_validate_explanation_rejects_contradictory_non_image_nchw_metadata(self) -> None:
        explanation = _explanation(
            input_kind="tabular",
            input_layout="NCHW",
            output_layout="NCHW",
            shape=(2, 3, 32, 32),
            method_families={MethodFamily.SHAPLEY, MethodFamily.GRADIENT},
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
            method_families={MethodFamily.SHAPLEY, MethodFamily.GRADIENT},
        )

        with pytest.raises(ValueError, match=r"ShapImageVisualiser.*input metadata"):
            ShapImageVisualiser().validate_explanation(explanation, object(), None)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "method_families",
        [
            {MethodFamily.SHAPLEY, MethodFamily.PERTURBATION},
            {MethodFamily.SHAPLEY, MethodFamily.TREE},
        ],
    )
    def test_validate_explanation_rejects_non_gradient_shap_semantics(
        self,
        method_families: AbstractSet[MethodFamily],
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
    def test_can_disable_original_image_panel_with_neutral_runtime_kwarg(
        self,
        sample_images: torch.Tensor,
    ) -> None:
        visualiser = ShapImageVisualiser()
        attributions = torch.randn_like(sample_images)

        fig = visualiser.visualise(
            attributions,
            inputs=sample_images,
            max_samples=1,
            include_original_input=False,
        )

        assert "Original Image" not in [ax.get_title() for ax in fig.axes]
        plt.close(fig)

    @pytest.mark.usefixtures("needs_shap")
    def test_runtime_include_original_image_alias_warns(
        self,
        sample_images: torch.Tensor,
    ) -> None:
        visualiser = ShapImageVisualiser()
        attributions = torch.randn_like(sample_images)

        with pytest.warns(DeprecationWarning, match="include_original_input"):
            fig = visualiser.visualise(
                attributions,
                inputs=sample_images,
                max_samples=1,
                include_original_image=False,
            )

        assert "Original Image" not in [ax.get_title() for ax in fig.axes]
        plt.close(fig)

    @pytest.mark.usefixtures("needs_shap")
    def test_neutral_runtime_kwarg_wins_over_deprecated_alias_without_warning(
        self,
        sample_images: torch.Tensor,
    ) -> None:
        visualiser = ShapImageVisualiser()
        attributions = torch.randn_like(sample_images)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig = visualiser.visualise(
                attributions,
                inputs=sample_images,
                max_samples=1,
                include_original_input=False,
                include_original_image=True,
            )

        assert caught == []
        assert "Original Image" not in [ax.get_title() for ax in fig.axes]
        plt.close(fig)

    @pytest.mark.usefixtures("needs_shap")
    def test_red_transparent_blue_accessor_returns_shap_colormap(self) -> None:
        """``_red_transparent_blue`` lazily returns SHAP's diverging colormap."""
        from matplotlib.colors import Colormap

        from raitap.transparency.visualisers.shap_visualisers import _red_transparent_blue

        cmap = _red_transparent_blue()
        assert isinstance(cmap, Colormap)
        assert cmap.name == "red_transparent_blue"

    def test_rgb_to_grayscale_uses_luminosity_weights(self) -> None:
        """RGB → 2-D grayscale via ITU-R luminosity weights matching shap.image_plot."""
        from raitap.transparency.visualisers.shap_visualisers import _rgb_to_grayscale

        # Pure red, green, blue (H=W=2). Channels last (H, W, C).
        red = np.zeros((2, 2, 3))
        red[..., 0] = 1.0
        green = np.zeros((2, 2, 3))
        green[..., 1] = 1.0
        blue = np.zeros((2, 2, 3))
        blue[..., 2] = 1.0

        np.testing.assert_allclose(_rgb_to_grayscale(red), np.full((2, 2), 0.2989))
        np.testing.assert_allclose(_rgb_to_grayscale(green), np.full((2, 2), 0.5870))
        np.testing.assert_allclose(_rgb_to_grayscale(blue), np.full((2, 2), 0.1140))

    def test_rgb_to_grayscale_passthrough_and_non_rgb_mean(self) -> None:
        """2-D inputs pass through, single-channel reduces to 2-D, non-RGB averages."""
        from raitap.transparency.visualisers.shap_visualisers import _rgb_to_grayscale

        # 2-D pass-through.
        flat = np.arange(4, dtype=float).reshape(2, 2)
        np.testing.assert_array_equal(_rgb_to_grayscale(flat), flat)

        # Single-channel HxWx1 — fall back to mean (one channel = identity).
        single = np.full((2, 2, 1), 0.7)
        np.testing.assert_allclose(_rgb_to_grayscale(single), np.full((2, 2), 0.7))

        # 5-channel multi-channel non-RGB — per-channel mean.
        multi = np.stack([np.full((2, 2), float(c)) for c in range(5)], axis=-1)
        np.testing.assert_allclose(_rgb_to_grayscale(multi), np.full((2, 2), 2.0))

    def test_rgb_to_grayscale_rejects_unsupported_shapes(self) -> None:
        from raitap.transparency.visualisers.shap_visualisers import _rgb_to_grayscale

        with pytest.raises(ValueError, match=r"expected 2D or 3D"):
            _rgb_to_grayscale(np.zeros((1, 2, 3, 4)))

    def test_symmetric_vmin_vmax_uses_nanpercentile(self) -> None:
        """vmax = nanpercentile(|values|, perc); vmin = -vmax."""
        from raitap.transparency.visualisers.shap_visualisers import _symmetric_vmin_vmax

        values = np.array([-3.0, -1.0, 0.0, 2.0, 5.0, np.nan])
        vmin, vmax = _symmetric_vmin_vmax(values, outlier_perc=99.9)
        expected = float(np.nanpercentile(np.abs(values), 99.9))
        assert vmax == pytest.approx(expected)
        assert vmin == pytest.approx(-expected)

    def test_symmetric_vmin_vmax_falls_back_for_all_zero(self) -> None:
        """All-zero / empty / non-finite inputs fall back to ±1.0."""
        from raitap.transparency.visualisers.shap_visualisers import _symmetric_vmin_vmax

        assert _symmetric_vmin_vmax(np.zeros((4, 4))) == (-1.0, 1.0)
        assert _symmetric_vmin_vmax(np.array([])) == (-1.0, 1.0)
        assert _symmetric_vmin_vmax(np.full((2, 2), np.nan)) == (-1.0, 1.0)

    def test_init_defaults_match_shap_image_plot(self) -> None:
        """Defaults match shap.plots.image: cmap None sentinel, alpha=0.15, outlier_perc=99.9."""
        from raitap.transparency.visualisers.shap_visualisers import ShapImageVisualiser

        v = ShapImageVisualiser()
        assert v.cmap is None  # resolved to SHAP's red_transparent_blue at render time
        assert v.overlay_alpha == 0.15
        assert v.outlier_perc == 99.9
        assert v.max_samples == 4
        assert v.include_original_image is True
        assert v.show_colorbar is True

    def test_init_accepts_explicit_cmap_and_outlier_perc(self) -> None:
        from raitap.transparency.visualisers.shap_visualisers import ShapImageVisualiser

        v = ShapImageVisualiser(cmap="viridis", outlier_perc=95.0, overlay_alpha=0.4)
        assert v.cmap == "viridis"
        assert v.outlier_perc == 95.0
        assert v.overlay_alpha == 0.4

    @pytest.mark.usefixtures("needs_shap")
    def test_visualise_uses_shap_native_recipe(self) -> None:
        """Each panel draws grayscale@overlay_alpha under red_transparent_blue@±perc."""
        import matplotlib

        matplotlib.use("Agg")

        from raitap.transparency.visualisers.shap_visualisers import (
            ShapImageVisualiser,
            _image_heatmap,
            _symmetric_vmin_vmax,
        )

        rng = np.random.default_rng(0)
        # 2 samples, RGB 8x8.
        attributions = torch.from_numpy(rng.normal(size=(2, 3, 8, 8))).float()
        inputs = torch.from_numpy(rng.uniform(size=(2, 3, 8, 8))).float()

        fig = ShapImageVisualiser(include_original_image=False, show_colorbar=False).visualise(
            attributions, inputs=inputs
        )

        # First sample's attribution axis is axes[0] (no original, no colorbar).
        attr_ax = fig.axes[0]
        images = attr_ax.get_images()
        assert len(images) == 2, "expected grayscale background + heatmap overlay"

        bg, heat = images
        assert bg.cmap.name == "gray"
        assert bg.get_alpha() == pytest.approx(0.15)

        assert heat.cmap.name == "red_transparent_blue"
        vmin, vmax = heat.get_clim()
        # Compute the expected percentile from the same heatmap reduction the
        # visualiser uses internally.
        heatmap_first = _image_heatmap(np.transpose(attributions[0].numpy(), (1, 2, 0)))
        expected_vmin, expected_vmax = _symmetric_vmin_vmax(heatmap_first, 99.9)
        assert vmin == pytest.approx(expected_vmin)
        assert vmax == pytest.approx(expected_vmax)

    @pytest.mark.usefixtures("needs_shap")
    def test_visualise_without_inputs_skips_grayscale_background(self) -> None:
        """When no input image is provided the grayscale background is omitted."""
        import matplotlib

        matplotlib.use("Agg")

        from raitap.transparency.visualisers.shap_visualisers import ShapImageVisualiser

        attributions = torch.zeros(1, 3, 8, 8)
        attributions[..., 0, 0] = 1.0  # nonzero so the colormap range is valid

        fig = ShapImageVisualiser(include_original_image=False, show_colorbar=False).visualise(
            attributions, inputs=None
        )

        attr_ax = fig.axes[0]
        images = attr_ax.get_images()
        # Only the heatmap — no grayscale background without an input image.
        assert len(images) == 1
        assert images[0].cmap.name == "red_transparent_blue"


class TestInputThumbnailVisualiser:
    def test_renders_single_image_input_without_original_embedding_contract(self) -> None:
        visualiser = InputThumbnailVisualiser()
        explanation = _explanation(
            input_kind="image",
            input_layout="NCHW",
            output_layout="NCHW",
            shape=(1, 3, 8, 8),
        )

        visualiser.validate_explanation(
            explanation,
            torch.zeros(1, 3, 8, 8),
            torch.rand(1, 3, 8, 8),
        )
        fig = visualiser.visualise(
            torch.zeros(1, 3, 8, 8),
            inputs=torch.rand(1, 3, 8, 8),
            max_samples=1,
        )

        assert type(visualiser).embeds_original_input is False
        assert len(fig.axes) == 1
        assert fig.axes[0].get_title() == "Input"
        plt.close(fig)

    def test_rejects_unsupported_input_kind(self) -> None:
        visualiser = InputThumbnailVisualiser()

        with pytest.raises(ValueError, match=r"InputThumbnailVisualiser.*input metadata"):
            visualiser.validate_explanation(
                _explanation(input_kind="tabular", shape=(1, 4)),
                torch.zeros(1, 4),
                torch.zeros(1, 4),
            )

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


class _DeltaVisualiser(BaseVisualiser):
    supported_structured_payload_kinds = frozenset({StructuredPayloadKind.CONVERGENCE_DELTA})

    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Figure:
        del attributions, inputs, kwargs
        fig, _ax = plt.subplots(figsize=(1, 1))
        return fig


def _explanation_with_payloads(payloads: list[StructuredPayload]) -> SimpleNamespace:
    base = _explanation()
    base.structured_payloads = payloads
    return base


def test_declared_structured_kind_accepts_matching_payload() -> None:
    explanation = _explanation_with_payloads(
        [StructuredPayload("convergence_delta", StructuredPayloadKind.CONVERGENCE_DELTA, None)]
    )
    _DeltaVisualiser().validate_explanation(explanation, torch.zeros(4, 10), None)


def test_declared_structured_kind_rejects_missing_payload() -> None:
    explanation = _explanation_with_payloads([])
    with pytest.raises(ValueError, match=r"structured payload"):
        _DeltaVisualiser().validate_explanation(explanation, torch.zeros(4, 10), None)


def test_default_visualiser_ignores_structured_payloads() -> None:
    # A visualiser declaring nothing must accept an explanation regardless of payloads.
    explanation = _explanation_with_payloads(
        [StructuredPayload("convergence_delta", StructuredPayloadKind.CONVERGENCE_DELTA, None)]
    )
    _ContractVisualiser().validate_explanation(explanation, torch.zeros(4, 10), None)


class _MultiKindVisualiser(BaseVisualiser):
    supported_structured_payload_kinds = frozenset(
        {StructuredPayloadKind.CONVERGENCE_DELTA, StructuredPayloadKind.BASE_VALUE}
    )

    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Figure:
        del attributions, inputs, kwargs
        fig, _ax = plt.subplots(figsize=(1, 1))
        return fig


def test_declared_structured_kinds_accept_any_one_present() -> None:
    # Any-of: a visualiser declaring multiple kinds is compatible when the
    # explanation carries at least one of them (no explainer emits every kind).
    explanation = _explanation_with_payloads(
        [StructuredPayload("base_value", StructuredPayloadKind.BASE_VALUE, None)]
    )
    _MultiKindVisualiser().validate_explanation(explanation, torch.zeros(4, 10), None)


def test_detection_uses_shap_renderer_for_shap_library(monkeypatch: pytest.MonkeyPatch) -> None:
    import raitap.transparency.visualisers.image_rendering as ir
    from raitap.transparency.contracts import DetectionBox, MethodFamily, VisualisationContext
    from raitap.transparency.visualisers.detection_image_visualiser import DetectionImageVisualiser

    calls: dict[str, bool] = {}

    class _Spy:
        def draw(
            self,
            ax: Any,
            attr: Any,
            image: Any,
            *,
            sign: str = "all",
            **style: Any,
        ) -> None:
            calls["used"] = True
            return ax.imshow(np.zeros((4, 4)))

    monkeypatch.setitem(ir.IMAGE_RENDERER_REGISTRY, "shap", _Spy())

    attr = torch.zeros(3, 4, 4)
    img = torch.zeros(3, 4, 4)
    ctx = VisualisationContext(
        algorithm="GradientExplainer",
        sample_names=None,
        show_sample_names=False,
        detection_box=DetectionBox(0, 0, (0.0, 0.0, 2.0, 2.0), 0.9, 1, "cat"),
        source_library="shap",
        method_families={MethodFamily.SHAPLEY},
    )
    fig = DetectionImageVisualiser().visualise(attr, inputs=img, context=ctx)
    assert calls.get("used") is True
    assert all(im.get_cmap().name != "seismic" for ax in fig.axes for im in ax.images)
    plt.close(fig)


@pytest.mark.parametrize(
    "visualiser_cls",
    [ShapBarVisualiser, ShapBeeswarmVisualiser, TabularBarChartVisualiser],
)
def test_feature_names_from_config_coerced_to_plain_list(visualiser_cls: type) -> None:
    # From YAML, feature_names arrives as an OmegaConf ListConfig, which
    # shap.summary_plot cannot index with numpy int64. The visualisers must
    # coerce it to a plain list at construction.
    from omegaconf import ListConfig, OmegaConf

    names = OmegaConf.create(["f0", "f1", "f2"])
    assert isinstance(names, ListConfig)
    vis = visualiser_cls(feature_names=names)
    assert type(vis.feature_names) is list
    assert vis.feature_names == ["f0", "f1", "f2"]
