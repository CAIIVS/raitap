from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from raitap.configs import set_output_root
from raitap.configs.schema import AppConfig, ReportingConfig
from raitap.reporting.builder import (
    BuiltReport,
    _canonical_facet_owners,
    _copy_asset,
    _render_kwargs_for_robustness_visualiser,
    build_merged_report,
    build_report,
)
from raitap.reporting.factory import create_report
from raitap.reporting.hydra_callback import ReportingSweepCallback
from raitap.reporting.manifest import ReportManifest
from raitap.reporting.sections import ReportGroup, ReportSection
from raitap.robustness.contracts import (
    MethodKind,
    Objective,
    PerturbationBudget,
    PerturbationNorm,
    RobustnessSemantics,
    RobustnessVerdict,
    RobustnessVisualisationContext,
    ThreatModel,
)
from raitap.robustness.results import (
    ConfiguredRobustnessVisualiser,
    RobustnessMetrics,
    RobustnessResult,
    RobustnessVisualisationResult,
    encode_verdicts,
)
from raitap.robustness.visualisers import ImagePairVisualiser, PerturbationHeatmapVisualiser
from raitap.robustness.visualisers.base_visualiser import BaseRobustnessVisualiser
from raitap.run.outputs import PredictionSummary, RunOutputs
from raitap.transparency.contracts import (
    ExplanationOutputSpace,
    ExplanationPayloadKind,
    ExplanationScope,
    ExplanationSemantics,
    ExplanationTarget,
    InputSpec,
    MethodFamily,
    OutputSpaceSpec,
    ScopeDefinitionStep,
)
from raitap.transparency.results import ConfiguredVisualiser, ExplanationResult, VisualisationResult
from raitap.transparency.visualisers.base_visualiser import BaseVisualiser

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class _MetricsStub:
    def __init__(self, image_path: Path) -> None:
        self._image_path = image_path

    def to_report_group(self) -> ReportGroup:
        return ReportGroup(
            heading="Performance Metrics",
            images=(self._image_path,),
            table_rows=(("accuracy", "0.9000"),),
        )


class _LocalImageVisualiser(BaseVisualiser):
    def __init__(
        self,
        *,
        title: str | None = None,
        method: str | None = None,
        sign: str | None = None,
        show_colorbar: bool | None = None,
    ) -> None:
        self.figures: list[Figure] = []
        self.calls: list[dict[str, Any]] = []
        self.title = title
        self.method = method
        self.sign = sign
        self.show_colorbar = show_colorbar

    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        *,
        context: Any = None,
        **kwargs: Any,
    ) -> Figure:
        del attributions, inputs
        self.calls.append(dict(kwargs))
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow([[0.0, 1.0], [1.0, 0.0]], cmap="magma")
        if context is not None and context.sample_names:
            ax.set_title(str(context.sample_names[0]))
        ax.axis("off")
        self.figures.append(fig)
        return fig


class _EmbeddedOriginalVisualiser(_LocalImageVisualiser):
    embeds_original_input = True


class _MaskedLikeVisualiser(_LocalImageVisualiser):
    embeds_original_input = True

    def renders_attribution_only_when_original_hidden(self) -> bool:
        return False


class _RobustnessRecordingVisualiser(BaseRobustnessVisualiser):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def visualise(
        self,
        result: RobustnessResult,
        *,
        context: RobustnessVisualisationContext,
        **kwargs: Any,
    ) -> Figure:
        del result, context
        self.calls.append(dict(kwargs))
        fig, _ax = plt.subplots(figsize=(1, 1))
        return fig


class _PerturbationRecordingVisualiser(_RobustnessRecordingVisualiser):
    embeds_perturbation_map = True


class _StrictPerturbationVisualiser(BaseRobustnessVisualiser):
    embeds_perturbation_map = True

    def visualise(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        result: RobustnessResult,
        *,
        context: RobustnessVisualisationContext,
    ) -> Figure:
        del result, context
        fig, _ax = plt.subplots(figsize=(1, 1))
        return fig


def _local_image_semantics(
    shape: tuple[int, ...],
    *,
    target: int | str | list[int] | tuple[int, ...] | None = None,
    method_families: frozenset[MethodFamily] = frozenset({MethodFamily.GRADIENT}),
    output_space: ExplanationOutputSpace = ExplanationOutputSpace.INPUT_FEATURES,
    output_shape: tuple[int, ...] | None = None,
    layer_path: str | None = None,
    requires_interpolation: bool = False,
) -> ExplanationSemantics:
    return ExplanationSemantics(
        scope=ExplanationScope.LOCAL,
        scope_definition_step=ScopeDefinitionStep.EXPLAINER_OUTPUT,
        payload_kind=ExplanationPayloadKind.ATTRIBUTIONS,
        method_families=method_families,
        target=ExplanationTarget(target=target) if target is not None else None,
        sample_selection=None,
        input_spec=InputSpec(
            kind="image",
            shape=shape,
            layout="NCHW",
            metadata={"kind": "image", "layout": "NCHW"},
        ),
        output_space=OutputSpaceSpec(
            space=output_space,
            shape=shape if output_shape is None else output_shape,
            layout="NCHW",
            layer_path=layer_path,
            requires_interpolation=requires_interpolation,
        ),
    )


def _local_tabular_semantics(shape: tuple[int, ...]) -> ExplanationSemantics:
    return ExplanationSemantics(
        scope=ExplanationScope.LOCAL,
        scope_definition_step=ScopeDefinitionStep.EXPLAINER_OUTPUT,
        payload_kind=ExplanationPayloadKind.ATTRIBUTIONS,
        method_families=frozenset({MethodFamily.GRADIENT}),
        target=None,
        sample_selection=None,
        input_spec=InputSpec(
            kind="tabular",
            shape=shape,
            layout="(B,F)",
            metadata={"kind": "tabular", "layout": "(B,F)"},
        ),
        output_space=OutputSpaceSpec(
            space=ExplanationOutputSpace.INPUT_FEATURES,
            shape=shape,
            layout="(B,F)",
        ),
    )


def _robustness_semantics() -> RobustnessSemantics:
    return RobustnessSemantics(
        method_kind=MethodKind.EMPIRICAL_ATTACK,
        threat_model=ThreatModel.WHITE_BOX,
        objective=Objective.UNTARGETED,
        families=frozenset({"gradient_sign"}),
        budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.03),
        input_spec=InputSpec(
            kind="image",
            shape=(1, 3, 4, 4),
            layout="NCHW",
            metadata={"kind": "image", "layout": "NCHW"},
        ),
    )


def _make_robustness_result(
    tmp_path: Path,
    *,
    assessor_name: str = "fgsm",
    batch_size: int = 1,
    visualisers: list[ConfiguredRobustnessVisualiser] | None = None,
) -> RobustnessResult:
    clean = torch.rand(batch_size, 3, 4, 4)
    return RobustnessResult(
        clean_inputs=clean,
        targets=torch.arange(batch_size),
        clean_predictions=torch.arange(batch_size),
        verdicts=encode_verdicts([RobustnessVerdict.ATTACKED] * batch_size),
        metrics=RobustnessMetrics(
            clean_accuracy=1.0,
            adversarial_accuracy=0.0,
            attack_success_rate=1.0,
        ),
        run_dir=tmp_path / "robustness" / assessor_name,
        experiment_name="robustness",
        assessor_target="t",
        algorithm="FGSM",
        assessor_name=assessor_name,
        perturbed_inputs=(clean + 0.03).clamp(0.0, 1.0),
        perturbed_predictions=torch.arange(batch_size) + 1,
        perturbation_distance=torch.full((batch_size,), 0.03),
        semantics=_robustness_semantics(),
        visualisers=[] if visualisers is None else visualisers,
    )


def test_build_report_orders_sections_and_ranks_samples(tmp_path: Path) -> None:
    config = AppConfig(experiment_name="demo")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    metrics_image = _write_test_image(tmp_path / "metrics.png")
    explanation = ExplanationResult(
        attributions=torch.rand(3, 1, 4, 4),
        inputs=torch.rand(3, 1, 4, 4),
        run_dir=tmp_path / "transparency" / "exp",
        experiment_name="demo",
        explainer_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((3, 1, 4, 4)),
        explainer_name="captum_ig",
        kwargs={"sample_names": ["a", "b", "c"], "show_sample_names": True},
        visualisers=[ConfiguredVisualiser(visualiser=_LocalImageVisualiser())],
    )
    native_global_path = _write_test_image(tmp_path / "native_global.png")
    native_global = VisualisationResult(
        explanation=explanation,
        figure=plt.figure(),
        visualiser_name="Global_0",
        visualiser_target="test.Global_0",
        output_path=native_global_path,
        scope=ExplanationScope.GLOBAL,
    )

    outputs = RunOutputs(
        explanations=[explanation],
        visualisations=[native_global],
        metrics=_MetricsStub(metrics_image),  # type: ignore[arg-type]
        forward_output=torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.95, 0.05]]),
        sample_ids=["a", "b", "c"],
        prediction_summaries=(
            PredictionSummary(
                sample_index=0,
                sample_id="a",
                predicted_class=1,
                target_class=0,
                confidence=0.9,
                correct=False,
            ),
            PredictionSummary(
                sample_index=1,
                sample_id="b",
                predicted_class=0,
                target_class=0,
                confidence=0.2,
                correct=True,
            ),
            PredictionSummary(
                sample_index=2,
                sample_id="c",
                predicted_class=0,
                target_class=0,
                confidence=0.95,
                correct=True,
            ),
        ),
    )

    report = build_report(config, outputs)

    assert [section.title for section in report.sections] == [
        "Metrics",
        "Global Explanations",
        "Local Explanations",
    ]
    local_groups = report.sections[2].groups
    assert [group.metadata["role"] for group in local_groups] == [
        "sample_header",
        "local_visualiser",
        "sample_header",
        "local_visualiser",
        "sample_header",
        "local_visualiser",
    ]
    assert [group.metadata["sample_index"] for group in local_groups[::2]] == [0, 1, 2]
    assert [group.metadata["sample_index"] for group in local_groups[1::2]] == [0, 1, 2]
    assert local_groups[1].metadata["explainer_name"] == "captum_ig"
    assert local_groups[1].metadata["visualiser_index"] == 0
    assert [group.images[0].name for group in local_groups] == [
        "sample_0_thumbnail_0.png",
        "sample_0_captum_ig__LocalImageVisualiser_0.png",
        "sample_1_thumbnail_0.png",
        "sample_1_captum_ig__LocalImageVisualiser_0.png",
        "sample_2_thumbnail_0.png",
        "sample_2_captum_ig__LocalImageVisualiser_0.png",
    ]
    assert all(
        path.parent.name == "_assets"
        for section in report.sections
        for group in section.groups
        for path in group.images
    )


def test_copy_asset_rejects_path_like_target_names(tmp_path: Path) -> None:
    source = _write_test_image(tmp_path / "source.png")
    assets_dir = tmp_path / "reports" / "_assets"

    with pytest.raises(ValueError, match="simple filenames"):
        _copy_asset(source, assets_dir=assets_dir, target_name="../escape.png")


def test_build_report_skips_global_section_for_local_only_outputs(tmp_path: Path) -> None:
    config = AppConfig(experiment_name="local_only")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    explanation = ExplanationResult(
        attributions=torch.rand(3, 1, 4, 4),
        inputs=torch.rand(3, 1, 4, 4),
        run_dir=tmp_path / "transparency" / "exp",
        experiment_name="local_only",
        explainer_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((3, 1, 4, 4)),
        explainer_name="captum_ig",
        visualisers=[ConfiguredVisualiser(visualiser=_LocalImageVisualiser())],
    )
    outputs = RunOutputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.95, 0.05]]),
        prediction_summaries=(
            PredictionSummary(sample_index=0, predicted_class=1, confidence=0.9, correct=None),
            PredictionSummary(sample_index=1, predicted_class=0, confidence=0.8, correct=None),
            PredictionSummary(sample_index=2, predicted_class=0, confidence=0.95, correct=None),
        ),
    )

    report = build_report(config, outputs)

    assert [section.title for section in report.sections] == ["Local Explanations"]


def test_build_report_places_cohort_visualisations_between_global_and_local(
    tmp_path: Path,
) -> None:
    config = AppConfig(experiment_name="cohort")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    metrics_image = _write_test_image(tmp_path / "metrics.png")
    explanation = ExplanationResult(
        attributions=torch.rand(2, 1, 4, 4),
        inputs=torch.rand(2, 1, 4, 4),
        run_dir=tmp_path / "transparency" / "exp",
        experiment_name="cohort",
        explainer_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((2, 1, 4, 4)),
        explainer_name="captum_ig",
        visualisers=[ConfiguredVisualiser(visualiser=_LocalImageVisualiser())],
    )
    native_cohort_path = _write_test_image(tmp_path / "native_cohort.png")
    native_cohort = VisualisationResult(
        explanation=explanation,
        figure=plt.figure(),
        visualiser_name="Cohort_0",
        visualiser_target="test.Cohort_0",
        output_path=native_cohort_path,
        scope=ExplanationScope.COHORT,
    )
    outputs = RunOutputs(
        explanations=[explanation],
        visualisations=[native_cohort],
        metrics=_MetricsStub(metrics_image),  # type: ignore[arg-type]
        forward_output=torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
        prediction_summaries=(
            PredictionSummary(sample_index=0, predicted_class=1, confidence=0.9),
            PredictionSummary(sample_index=1, predicted_class=0, confidence=0.8),
        ),
    )

    report = build_report(config, outputs)

    assert [section.title for section in report.sections] == [
        "Metrics",
        "Cohort Explanations",
        "Local Explanations",
    ]
    assert report.sections[1].groups[0].metadata["role"] == "cohort"


def test_build_report_local_assets_are_staged_and_closed(tmp_path: Path) -> None:
    config = AppConfig(experiment_name="local_assets")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    visualiser = _LocalImageVisualiser()
    explanation = ExplanationResult(
        attributions=torch.rand(3, 1, 4, 4),
        inputs=torch.rand(3, 1, 4, 4),
        run_dir=tmp_path / "transparency" / "exp",
        experiment_name="local_assets",
        explainer_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((3, 1, 4, 4)),
        explainer_name="captum_ig",
        kwargs={"sample_names": ["a", "b", "c"], "show_sample_names": True},
        visualisers=[ConfiguredVisualiser(visualiser=visualiser)],
    )
    outputs = RunOutputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.95, 0.05]]),
        sample_ids=["a", "b", "c"],
        prediction_summaries=(
            PredictionSummary(sample_index=0, sample_id="a", predicted_class=1, confidence=0.9),
            PredictionSummary(sample_index=1, sample_id="b", predicted_class=0, confidence=0.2),
            PredictionSummary(sample_index=2, sample_id="c", predicted_class=0, confidence=0.95),
        ),
    )

    report = build_report(config, outputs)

    local_groups = report.sections[0].groups
    assert len(local_groups) == 4
    assert [group.metadata["role"] for group in local_groups] == [
        "sample_header",
        "local_visualiser",
        "sample_header",
        "local_visualiser",
    ]
    assert [group.metadata["sample_index"] for group in local_groups] == [1, 1, 2, 2]
    assert [group.images[0].name for group in local_groups] == [
        "sample_1_thumbnail_0.png",
        "sample_1_captum_ig__LocalImageVisualiser_0.png",
        "sample_2_thumbnail_0.png",
        "sample_2_captum_ig__LocalImageVisualiser_0.png",
    ]
    assert all(ax.get_title() == "" for fig in visualiser.figures for ax in fig.axes)
    assert all(not plt.fignum_exists(fig.number) for fig in visualiser.figures)


def test_build_report_compact_local_thumbnail_titles_are_stripped(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    config = AppConfig(experiment_name="compact_thumbnail_titles")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    thumbnail_figures: list[Figure] = []

    def _titled_thumbnail(*_args: Any, **_kwargs: Any) -> Figure:
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.imshow([[0.0, 1.0], [1.0, 0.0]])
        ax.set_title("Input: ISIC_0001")
        fig.suptitle("Input thumbnail")
        thumbnail_figures.append(fig)
        return fig

    monkeypatch.setattr(
        "raitap.reporting.builder.InputThumbnailVisualiser.visualise",
        _titled_thumbnail,
    )

    explanation = ExplanationResult(
        attributions=torch.rand(1, 1, 4, 4),
        inputs=torch.rand(1, 1, 4, 4),
        run_dir=tmp_path / "transparency" / "exp",
        experiment_name="compact_thumbnail_titles",
        explainer_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((1, 1, 4, 4)),
        explainer_name="captum_ig",
        visualisers=[ConfiguredVisualiser(visualiser=_LocalImageVisualiser())],
    )
    outputs = RunOutputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=torch.tensor([[0.1, 0.9]]),
        prediction_summaries=(
            PredictionSummary(sample_index=0, predicted_class=1, confidence=0.9),
        ),
    )

    build_report(config, outputs)

    assert thumbnail_figures
    assert all(text.get_text() == "" for fig in thumbnail_figures for text in fig.texts)
    assert all(ax.get_title() == "" for fig in thumbnail_figures for ax in fig.axes)


def test_build_report_compact_mode_omits_repeated_original_for_capable_visualisers(
    tmp_path: Path,
) -> None:
    config = AppConfig(experiment_name="compact")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    compact_visualiser = _EmbeddedOriginalVisualiser()
    masked_visualiser = _MaskedLikeVisualiser()
    explanation = ExplanationResult(
        attributions=torch.rand(1, 1, 4, 4),
        inputs=torch.rand(1, 1, 4, 4),
        run_dir=tmp_path / "transparency" / "exp",
        experiment_name="compact",
        explainer_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((1, 1, 4, 4)),
        explainer_name="captum_ig",
        visualisers=[
            ConfiguredVisualiser(visualiser=compact_visualiser),
            ConfiguredVisualiser(visualiser=masked_visualiser),
        ],
    )
    outputs = RunOutputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=torch.tensor([[0.1, 0.9]]),
        prediction_summaries=(
            PredictionSummary(sample_index=0, predicted_class=1, confidence=0.9),
        ),
    )

    report = build_report(config, outputs)

    local_groups = report.sections[0].groups
    assert [group.metadata["role"] for group in local_groups] == [
        "sample_header",
        "local_visualiser",
        "local_visualiser",
    ]
    assert [group.metadata["visualiser_index"] for group in local_groups[1:]] == [0, 1]
    assert [len(group.images) for group in local_groups] == [1, 1, 1]
    assert compact_visualiser.calls == [{"include_original_input": False}]
    assert masked_visualiser.calls == [{}]


def test_build_report_local_explainer_group_includes_curated_transparency_rows(
    tmp_path: Path,
) -> None:
    config = AppConfig(experiment_name="transparency_rows")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    heatmap_visualiser = _LocalImageVisualiser(
        title="Grad-CAM lesion localisation",
        method="heat_map",
        sign="positive",
        show_colorbar=True,
    )
    masked_visualiser = _LocalImageVisualiser(
        title="Evidence-masked dermoscopy view",
        method="masked_image",
        sign="absolute_value",
        show_colorbar=False,
    )
    explanation = ExplanationResult(
        attributions=torch.rand(2, 1, 2, 2),
        inputs=torch.rand(2, 3, 4, 4),
        run_dir=tmp_path / "transparency" / "gradcam",
        experiment_name="transparency_rows",
        explainer_target="raitap.transparency.CaptumExplainer",
        algorithm="LayerGradCam",
        semantics=_local_image_semantics(
            (2, 3, 4, 4),
            target=[5, 6],
            method_families=frozenset({MethodFamily.CAM, MethodFamily.GRADIENT}),
            output_space=ExplanationOutputSpace.IMAGE_SPATIAL_MAP,
            output_shape=(2, 1, 2, 2),
            layer_path="1.layer4.2.conv3",
            requires_interpolation=True,
        ),
        explainer_name="gradcam_localisation",
        call_kwargs={"relu_attributions": True},
        visualisers=[
            ConfiguredVisualiser(
                visualiser=heatmap_visualiser,
                call_kwargs={"max_samples": 4},
            ),
            ConfiguredVisualiser(
                visualiser=masked_visualiser,
                call_kwargs={"max_samples": 1},
            ),
        ],
    )
    outputs = RunOutputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
        prediction_summaries=(
            PredictionSummary(sample_index=0, predicted_class=1, confidence=0.9),
            PredictionSummary(sample_index=1, predicted_class=0, confidence=0.8),
        ),
    )

    report = build_report(config, outputs)

    local_groups = report.sections[0].groups
    visualiser_groups = [
        group for group in local_groups if group.metadata["role"] == "local_visualiser"
    ]
    assert len(visualiser_groups) == 4
    assert visualiser_groups[0].heading == (
        "Explainer: gradcam_localisation - Visualiser: Grad-CAM lesion localisation"
    )
    assert visualiser_groups[1].heading == (
        "Explainer: gradcam_localisation - Visualiser: Evidence-masked dermoscopy view"
    )

    rows = dict(visualiser_groups[2].table_rows)
    assert rows["explainer"] == "gradcam_localisation"
    assert rows["algorithm"] == "LayerGradCam"
    assert rows["method_families"] == "cam, gradient"
    assert rows["targets"] == "0: 5"
    assert rows["output_space"] == "image_spatial_map"
    assert rows["output_shape"] == "2 x 1 x 2 x 2"
    assert rows["layer_path"] == "1.layer4.2.conv3"
    assert rows["requires_interpolation"] == "true"
    assert rows["call.relu_attributions"] == "true"
    assert rows["visualiser_title"] == "Grad-CAM lesion localisation"
    assert rows["visualiser_method"] == "heat_map"
    assert rows["visualiser_sign"] == "positive"
    assert "visualiser_show_colorbar" not in rows
    assert "visualiser_colorbar" not in rows
    assert "visualiser_include_original_image" not in rows
    assert "visualiser_embeds_original_input" not in rows
    assert "visualiser_attribution_only_without_original" not in rows
    assert "visualiser_call.max_samples" not in rows
    assert "visualiser_1_title" not in rows

    second_rows = dict(visualiser_groups[3].table_rows)
    assert second_rows["visualiser_title"] == "Evidence-masked dermoscopy view"
    assert second_rows["visualiser_method"] == "masked_image"
    assert "visualiser_show_colorbar" not in second_rows
    assert "visualiser_call.max_samples" not in second_rows


def test_build_report_show_original_per_explainer_uses_legacy_local_layout(
    tmp_path: Path,
) -> None:
    config = AppConfig(experiment_name="legacy_originals")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(
        _target_="PDFReporter",
        filename="report.pdf",
        show_original_per_explainer=True,
    )

    visualiser = _EmbeddedOriginalVisualiser()
    explanation = ExplanationResult(
        attributions=torch.rand(2, 1, 4, 4),
        inputs=torch.rand(2, 1, 4, 4),
        run_dir=tmp_path / "transparency" / "exp",
        experiment_name="legacy_originals",
        explainer_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((2, 1, 4, 4)),
        explainer_name="captum_ig",
        kwargs={"sample_names": ["legacy-a", "legacy-b"], "show_sample_names": True},
        visualisers=[ConfiguredVisualiser(visualiser=visualiser)],
    )
    outputs = RunOutputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
        prediction_summaries=(
            PredictionSummary(sample_index=0, predicted_class=1, confidence=0.9),
            PredictionSummary(sample_index=1, predicted_class=0, confidence=0.8),
        ),
    )

    report = build_report(config, outputs)

    local_groups = report.sections[0].groups
    assert len(local_groups) == 2
    assert all(group.metadata["role"] != "sample_header" for group in local_groups)
    assert all("include_original_input" not in call for call in visualiser.calls)
    assert any(ax.get_title() for fig in visualiser.figures for ax in fig.axes)


def test_build_report_thumbnail_uses_first_compatible_explanation_in_order(
    tmp_path: Path,
) -> None:
    config = AppConfig(experiment_name="thumbnail_fallback")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    image_visualiser = _EmbeddedOriginalVisualiser()
    tabular_visualiser = _EmbeddedOriginalVisualiser()
    image_explanation = ExplanationResult(
        attributions=torch.rand(1, 1, 4, 4),
        inputs=torch.rand(1, 1, 4, 4),
        run_dir=tmp_path / "transparency" / "image",
        experiment_name="thumbnail_fallback",
        explainer_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((1, 1, 4, 4)),
        explainer_name="image_exp",
        visualisers=[ConfiguredVisualiser(visualiser=image_visualiser)],
    )
    tabular_explanation = ExplanationResult(
        attributions=torch.rand(1, 4),
        inputs=torch.rand(1, 4),
        run_dir=tmp_path / "transparency" / "tabular",
        experiment_name="thumbnail_fallback",
        explainer_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_tabular_semantics((1, 4)),
        explainer_name="tabular_exp",
        visualisers=[ConfiguredVisualiser(visualiser=tabular_visualiser)],
    )
    outputs = RunOutputs(
        explanations=[tabular_explanation, image_explanation],
        visualisations=[],
        metrics=None,
        forward_output=torch.tensor([[0.1, 0.9]]),
        prediction_summaries=(
            PredictionSummary(sample_index=0, predicted_class=1, confidence=0.9),
        ),
    )

    report = build_report(config, outputs)

    local_groups = report.sections[0].groups
    assert [group.metadata["role"] for group in local_groups] == [
        "sample_header",
        "local_visualiser",
        "local_visualiser",
    ]
    assert local_groups[0].metadata["source_explainer_name"] == "image_exp"
    assert local_groups[1].metadata["explainer_name"] == "tabular_exp"
    assert local_groups[1].metadata["thumbnail_source_explainer_names"] == ("image_exp",)
    assert local_groups[2].metadata["explainer_name"] == "image_exp"
    assert local_groups[2].metadata["thumbnail_source_explainer_names"] == ("image_exp",)
    assert tabular_visualiser.calls == [{"include_original_input": False}]
    assert image_visualiser.calls == [{"include_original_input": False}]


def test_build_report_thumbnail_failure_falls_back_for_that_sample_only(
    tmp_path: Path,
) -> None:
    config = AppConfig(experiment_name="thumbnail_fallback")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    visualiser = _EmbeddedOriginalVisualiser()
    explanation = ExplanationResult(
        attributions=torch.rand(1, 4),
        inputs=torch.rand(1, 4),
        run_dir=tmp_path / "transparency" / "tabular",
        experiment_name="thumbnail_fallback",
        explainer_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_tabular_semantics((1, 4)),
        explainer_name="tabular_exp",
        visualisers=[ConfiguredVisualiser(visualiser=visualiser)],
    )
    outputs = RunOutputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=torch.tensor([[0.1, 0.9]]),
        prediction_summaries=(
            PredictionSummary(sample_index=0, predicted_class=1, confidence=0.9),
        ),
    )

    report = build_report(config, outputs)

    local_groups = report.sections[0].groups
    assert len(local_groups) == 1
    assert local_groups[0].metadata["role"] == "local_visualiser"
    assert visualiser.calls == [{}]


def test_build_report_thumbnail_runtime_failure_logs_traceback_and_falls_back(
    tmp_path: Path,
    monkeypatch: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = AppConfig(experiment_name="thumbnail_runtime_fallback")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    def _raise_runtime_error(*_args: Any, **_kwargs: Any) -> Figure:
        raise RuntimeError("thumbnail render failed")

    monkeypatch.setattr(
        "raitap.reporting.builder.InputThumbnailVisualiser.visualise",
        _raise_runtime_error,
    )

    visualiser = _EmbeddedOriginalVisualiser()
    explanation = ExplanationResult(
        attributions=torch.rand(1, 1, 4, 4),
        inputs=torch.rand(1, 1, 4, 4),
        run_dir=tmp_path / "transparency" / "exp",
        experiment_name="thumbnail_runtime_fallback",
        explainer_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((1, 1, 4, 4)),
        explainer_name="captum_ig",
        visualisers=[ConfiguredVisualiser(visualiser=visualiser)],
    )
    outputs = RunOutputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=torch.tensor([[0.1, 0.9]]),
        prediction_summaries=(
            PredictionSummary(sample_index=0, predicted_class=1, confidence=0.9),
        ),
    )

    with caplog.at_level("WARNING"):
        report = build_report(config, outputs)

    local_groups = report.sections[0].groups
    assert len(local_groups) == 1
    assert local_groups[0].metadata["role"] == "local_visualiser"
    assert visualiser.calls == [{}]
    assert any(
        record.exc_info is not None and "thumbnail render failed" in record.getMessage()
        for record in caplog.records
    )


def test_build_report_thumbnail_programmer_error_is_not_swallowed(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    config = AppConfig(experiment_name="thumbnail_programmer_error")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    def _raise_type_error(*_args: Any, **_kwargs: Any) -> Figure:
        raise TypeError("programmer error")

    monkeypatch.setattr(
        "raitap.reporting.builder.InputThumbnailVisualiser.visualise",
        _raise_type_error,
    )

    explanation = ExplanationResult(
        attributions=torch.rand(1, 1, 4, 4),
        inputs=torch.rand(1, 1, 4, 4),
        run_dir=tmp_path / "transparency" / "exp",
        experiment_name="thumbnail_programmer_error",
        explainer_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((1, 1, 4, 4)),
        explainer_name="captum_ig",
        visualisers=[ConfiguredVisualiser(visualiser=_EmbeddedOriginalVisualiser())],
    )
    outputs = RunOutputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=torch.tensor([[0.1, 0.9]]),
        prediction_summaries=(
            PredictionSummary(sample_index=0, predicted_class=1, confidence=0.9),
        ),
    )

    with pytest.raises(TypeError, match="programmer error"):
        build_report(config, outputs)


def test_build_report_skips_local_groups_when_no_local_visualisations(tmp_path: Path) -> None:
    config = AppConfig(experiment_name="no_local")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    class _GlobalOnlyVisualiser(_LocalImageVisualiser):
        produces_scope = ExplanationScope.GLOBAL

    explanation = ExplanationResult(
        attributions=torch.rand(2, 1, 4, 4),
        inputs=torch.rand(2, 1, 4, 4),
        run_dir=tmp_path / "transparency" / "exp",
        experiment_name="no_local",
        explainer_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((2, 1, 4, 4)),
        visualisers=[ConfiguredVisualiser(visualiser=_GlobalOnlyVisualiser())],
    )
    outputs = RunOutputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
        prediction_summaries=(
            PredictionSummary(sample_index=0, predicted_class=1, confidence=0.9),
            PredictionSummary(sample_index=1, predicted_class=0, confidence=0.8),
        ),
    )

    report = build_report(config, outputs)

    assert report.sections == ()


def test_canonical_facet_owners_prefers_dedicated_visualiser_in_any_order() -> None:
    pair = ConfiguredRobustnessVisualiser(visualiser=ImagePairVisualiser(max_samples=1))
    heatmap = ConfiguredRobustnessVisualiser(
        visualiser=PerturbationHeatmapVisualiser(max_samples=1)
    )

    assert _canonical_facet_owners([pair, heatmap])["perturbation_map"] == 1
    assert _canonical_facet_owners([heatmap, pair])["perturbation_map"] == 0
    assert _canonical_facet_owners([pair, heatmap])["clean_input"] == 0


def test_robustness_render_kwargs_raise_if_both_facets_would_be_omitted() -> None:
    visualiser = ImagePairVisualiser(max_samples=1)

    with pytest.raises(AssertionError, match="omit both"):
        _render_kwargs_for_robustness_visualiser(
            visualiser,
            owners={"clean_input": 1, "perturbation_map": 2},
            visualiser_index=0,
            omit_redundant=True,
        )


def test_robustness_no_embedder_receives_no_compact_kwargs() -> None:
    visualiser = _RobustnessRecordingVisualiser()
    owners = _canonical_facet_owners([ConfiguredRobustnessVisualiser(visualiser=visualiser)])

    assert owners == {}
    assert (
        _render_kwargs_for_robustness_visualiser(
            visualiser,
            owners=owners,
            visualiser_index=0,
            omit_redundant=True,
        )
        == {}
    )


def test_build_report_compact_robustness_omits_non_owner_perturbation_panel(
    tmp_path: Path,
) -> None:
    config = AppConfig(experiment_name="robustness_compact")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    result = _make_robustness_result(
        tmp_path,
        visualisers=[
            ConfiguredRobustnessVisualiser(visualiser=ImagePairVisualiser(max_samples=1)),
            ConfiguredRobustnessVisualiser(visualiser=PerturbationHeatmapVisualiser(max_samples=1)),
        ],
    )
    outputs = RunOutputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        forward_output=torch.zeros(1, 2),
        robustness_results=[result],
    )

    report = build_report(config, outputs)

    robustness_group = report.sections[0].groups[0]
    assert robustness_group.metadata["role"] == "robustness"
    assert len(robustness_group.images) == 2
    pair_image = plt.imread(robustness_group.images[0])
    heatmap_image = plt.imread(robustness_group.images[1])
    width_ratio = pair_image.shape[1] / heatmap_image.shape[1]
    assert 1.2 < width_ratio < 2.5
    assert dict(robustness_group.table_rows)["attack_success_rate"] == "1.0000"
    assert robustness_group.images[0].name.startswith(
        "robustness_0_fgsm_sample_0_ImagePairVisualiser_0"
    )


def test_build_report_compact_robustness_renders_selected_samples_per_assessor(
    tmp_path: Path,
) -> None:
    config = AppConfig(experiment_name="robustness_compact_samples")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    result = _make_robustness_result(
        tmp_path,
        batch_size=20,
        visualisers=[
            ConfiguredRobustnessVisualiser(visualiser=ImagePairVisualiser(max_samples=4)),
            ConfiguredRobustnessVisualiser(visualiser=PerturbationHeatmapVisualiser(max_samples=4)),
        ],
    )
    outputs = RunOutputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        forward_output=torch.zeros(20, 2),
        prediction_summaries=(
            PredictionSummary(
                sample_index=3,
                predicted_class=1,
                confidence=0.9,
                correct=False,
            ),
            PredictionSummary(
                sample_index=8,
                predicted_class=0,
                confidence=0.1,
                correct=False,
            ),
            PredictionSummary(
                sample_index=19,
                predicted_class=1,
                confidence=0.8,
                correct=True,
            ),
        ),
        robustness_results=[result],
    )

    report = build_report(config, outputs)

    robustness_group = report.sections[0].groups[0]
    assert robustness_group.metadata["role"] == "robustness"
    assert robustness_group.metadata["sample_indices"] == (3, 8, 19)
    assert len(report.sections[0].groups) == 1
    assert len(robustness_group.images) == 6
    image_names = [image.name for image in robustness_group.images]
    assert all("_sample_" in name for name in image_names)
    assert [name.split("_sample_", 1)[1].split("_", 1)[0] for name in image_names] == [
        "3",
        "3",
        "8",
        "8",
        "19",
        "19",
    ]
    assert any("sample_3_ImagePairVisualiser_0" in name for name in image_names)
    assert any("sample_8_PerturbationHeatmapVisualiser_1" in name for name in image_names)
    assert any("sample_19_ImagePairVisualiser_0" in name for name in image_names)


def test_build_report_robustness_single_pair_keeps_all_panels(tmp_path: Path) -> None:
    config = AppConfig(experiment_name="robustness_single_pair")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    result = _make_robustness_result(
        tmp_path,
        visualisers=[ConfiguredRobustnessVisualiser(visualiser=ImagePairVisualiser(max_samples=1))],
    )
    outputs = RunOutputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        forward_output=torch.zeros(1, 2),
        robustness_results=[result],
    )

    report = build_report(config, outputs)

    assert len(report.sections[0].groups[0].images) == 1


def test_build_report_legacy_robustness_reuses_existing_visualisations_without_kwargs(
    tmp_path: Path,
) -> None:
    config = AppConfig(experiment_name="robustness_legacy")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(
        _target_="PDFReporter",
        filename="report.pdf",
        show_redundant_robustness_panels=True,
    )

    recording = _PerturbationRecordingVisualiser()
    result = _make_robustness_result(
        tmp_path,
        visualisers=[ConfiguredRobustnessVisualiser(visualiser=recording)],
    )
    existing_image = _write_test_image(tmp_path / "robustness_existing.png")
    existing_visualisation = RobustnessVisualisationResult(
        result=result,
        figure=plt.figure(),
        visualiser_name="_RobustnessRecordingVisualiser_0",
        visualiser_target="test._RobustnessRecordingVisualiser_0",
        output_path=existing_image,
    )
    outputs = RunOutputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        forward_output=torch.zeros(1, 2),
        robustness_results=[result],
        robustness_visualisations=[existing_visualisation],
    )

    report = build_report(config, outputs)

    assert len(report.sections[0].groups[0].images) == 1
    assert recording.calls == []


def test_build_report_robustness_declared_facet_without_kwarg_support_surfaces_error(
    tmp_path: Path,
) -> None:
    config = AppConfig(experiment_name="robustness_contract_error")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    result = _make_robustness_result(
        tmp_path,
        visualisers=[
            ConfiguredRobustnessVisualiser(visualiser=PerturbationHeatmapVisualiser(max_samples=1)),
            ConfiguredRobustnessVisualiser(visualiser=_StrictPerturbationVisualiser()),
        ],
    )
    outputs = RunOutputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        forward_output=torch.zeros(1, 2),
        robustness_results=[result],
    )

    with pytest.raises(TypeError, match="include_perturbation_map"):
        build_report(config, outputs)


def test_report_manifest_round_trip_preserves_relative_images(tmp_path: Path) -> None:
    config = AppConfig(experiment_name="demo")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    explanation = ExplanationResult(
        attributions=torch.rand(2, 1, 4, 4),
        inputs=torch.rand(2, 1, 4, 4),
        run_dir=tmp_path / "transparency" / "exp",
        experiment_name="demo",
        explainer_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((2, 1, 4, 4)),
        explainer_name="captum_ig",
        visualisers=[ConfiguredVisualiser(visualiser=_LocalImageVisualiser())],
    )
    outputs = RunOutputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
        prediction_summaries=(
            PredictionSummary(sample_index=0, predicted_class=1, confidence=0.9),
            PredictionSummary(sample_index=1, predicted_class=0, confidence=0.8),
        ),
    )

    report = build_report(config, outputs)
    manifest_path = report.report_dir / "report_manifest.json"
    report.manifest.write(manifest_path, report_dir=report.report_dir)
    loaded = ReportManifest.load(manifest_path)

    assert [section.title for section in loaded.sections] == [
        section.title for section in report.sections
    ]
    assert all(
        path.exists()
        for section in loaded.sections
        for group in section.groups
        for path in group.images
    )


def test_report_manifest_rejects_paths_outside_report_dir(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    manifest_path = report_dir / "report_manifest.json"
    manifest_path.write_text(
        """
{
  "kind": "run",
  "filename": "report.pdf",
  "metadata": {},
  "sections": [
    {
      "title": "Metrics",
      "metadata": {},
      "groups": [
        {
          "heading": "Performance Metrics",
          "table_rows": [],
          "images": ["../outside.png"],
          "metadata": {}
        }
      ]
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes report directory"):
        ReportManifest.load(manifest_path)


def test_create_report_writes_manifest_next_to_generated_report(
    tmp_path: Path, monkeypatch: Any
) -> None:
    config = AppConfig(experiment_name="demo")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")
    built_report = ReportSection.from_groups(
        "Metrics",
        [ReportGroup(heading="Performance Metrics", table_rows=(("accuracy", "0.9"),))],
    )
    report = ReportManifest(kind="run", sections=(built_report,))
    built = BuiltReport(
        report_dir=tmp_path / "builder-reports",
        sections=(built_report,),
        manifest=report,
    )

    class _ReporterStub:
        def __init__(self, _config: Any) -> None:
            pass

        def generate(self, sections: Any, *, report_dir: Path | None = None) -> Path:
            del sections
            output_root = tmp_path / "generated-reports" if report_dir is None else report_dir
            output_path = output_root / "report.pdf"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"%PDF-1.4\n")
            return output_path

    monkeypatch.setattr("raitap.reporting.factory.instantiate", lambda _cfg: _ReporterStub)

    generated = create_report(config, built)

    assert generated.report_path.parent == built.report_dir
    assert generated.manifest_path == generated.report_path.parent / "report_manifest.json"
    assert generated.manifest_path.exists()


def test_build_report_manifest_filename_matches_selected_reporter(tmp_path: Path) -> None:
    outputs = RunOutputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        forward_output=torch.empty(0),
    )

    html_config = AppConfig(experiment_name="demo")
    set_output_root(html_config, tmp_path / "html")
    html_config.reporting = ReportingConfig(
        _target_="raitap.reporting.HTMLReporter",
        filename="custom_report.pdf",
    )
    html_report = build_report(html_config, outputs)

    assert html_report.manifest.filename == "custom_report.html"

    pdf_config = AppConfig(experiment_name="demo")
    set_output_root(pdf_config, tmp_path / "pdf")
    pdf_config.reporting = ReportingConfig(
        _target_="raitap.reporting.PDFReporter",
        filename="custom_report.pdf",
    )
    pdf_report = build_report(pdf_config, outputs)

    assert pdf_report.manifest.filename == "custom_report.pdf"


def test_reporting_sweep_callback_builds_merged_report_from_child_manifests(
    tmp_path: Path, monkeypatch: Any
) -> None:
    sweep_dir = tmp_path / "multirun"
    sweep_dir.mkdir()
    _write_child_manifest(sweep_dir / "0", heading="Metrics A")
    _write_child_manifest(sweep_dir / "1", heading="Metrics B")
    (sweep_dir / "2").mkdir()

    captured: dict[str, Any] = {}

    def _capture_report(cfg: Any, report: Any) -> Any:
        captured["config"] = cfg
        captured["report"] = report
        return SimpleNamespace(report_path=report.report_dir / "report.pdf")

    monkeypatch.setattr("raitap.reporting.hydra_callback.create_report", _capture_report)

    callback = ReportingSweepCallback()
    config = OmegaConf.create(
        {
            "experiment_name": "demo",
            "reporting": {"_target_": "PDFReporter", "filename": "report.pdf"},
            "hydra": {"sweep": {"dir": str(sweep_dir)}},
        }
    )
    callback.on_multirun_end(config)

    report = captured["report"]
    assert report.manifest.kind == "multirun"
    assert report.manifest.metadata["skipped_children"] == ["2"]
    assert [section.title for section in report.sections] == ["Metrics"]
    assert report.sections[0].groups[0].heading.startswith("Job 0")


def test_build_merged_report_deduplicates_identical_metrics_only(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "multirun"
    sweep_dir.mkdir()
    _write_child_manifest(
        sweep_dir / "0",
        heading="Metrics A",
        table_rows=(("accuracy", "0.9000"),),
        include_local=True,
    )
    _write_child_manifest(
        sweep_dir / "1",
        heading="Metrics B",
        table_rows=(("accuracy", "0.9000"),),
        include_local=True,
    )
    _write_child_manifest(
        sweep_dir / "2",
        heading="Metrics C",
        table_rows=(("accuracy", "0.8000"),),
        include_local=True,
    )
    child_manifests: list[tuple[str, str | None, ReportManifest]] = [
        (
            f"Job {index}",
            None,
            ReportManifest.load(sweep_dir / str(index) / "reports" / "report_manifest.json"),
        )
        for index in range(3)
    ]

    report = build_merged_report(
        AppConfig(experiment_name="demo"),
        sweep_dir=sweep_dir,
        child_manifests=child_manifests,
        skipped_children=[],
    )

    sections = {section.title: section for section in report.sections}
    assert [group.heading for group in sections["Metrics"].groups] == [
        "Job 0 - Metrics A",
        "Job 2 - Metrics C",
    ]
    assert len(sections["Local Explanations"].groups) == 3


def test_build_merged_report_preserves_present_section_order_with_cohort(
    tmp_path: Path,
) -> None:
    sweep_dir = tmp_path / "multirun"
    sweep_dir.mkdir()
    _write_child_manifest(
        sweep_dir / "0",
        heading="Metrics A",
        include_cohort=True,
        include_local=True,
    )
    child_manifests: list[tuple[str, str | None, ReportManifest]] = [
        (
            "Job 0",
            None,
            ReportManifest.load(sweep_dir / "0" / "reports" / "report_manifest.json"),
        )
    ]

    report = build_merged_report(
        AppConfig(experiment_name="demo"),
        sweep_dir=sweep_dir,
        child_manifests=child_manifests,
        skipped_children=[],
    )

    assert [section.title for section in report.sections] == [
        "Metrics",
        "Cohort Explanations",
        "Local Explanations",
    ]


def test_build_merged_report_keeps_empty_metrics_groups(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "multirun"
    sweep_dir.mkdir()
    _write_child_manifest(sweep_dir / "0", heading="Metrics A", table_rows=())
    _write_child_manifest(sweep_dir / "1", heading="Metrics B", table_rows=())
    child_manifests: list[tuple[str, str | None, ReportManifest]] = [
        (
            f"Job {index}",
            None,
            ReportManifest.load(sweep_dir / str(index) / "reports" / "report_manifest.json"),
        )
        for index in range(2)
    ]

    report = build_merged_report(
        AppConfig(experiment_name="demo"),
        sweep_dir=sweep_dir,
        child_manifests=child_manifests,
        skipped_children=[],
    )

    sections = {section.title: section for section in report.sections}
    assert [group.heading for group in sections["Metrics"].groups] == [
        "Job 0 - Metrics A",
        "Job 1 - Metrics B",
    ]


def test_reporting_configs_compose_multirun_report_controls() -> None:
    cfg = _compose_raitap_config()
    assert cfg.reporting._target_ == "HTMLReporter"
    assert cfg.reporting.multirun_report is True
    assert cfg.hydra.callbacks.reporting_sweep._target_.endswith("ReportingSweepCallback")

    disabled_cfg = _compose_raitap_config(["reporting=disabled"])
    assert disabled_cfg.reporting._target_ is None
    assert disabled_cfg.reporting.multirun_report is False
    assert disabled_cfg.hydra.get("callbacks") == {}

    pdf_cfg = _compose_raitap_config(["reporting=pdf"])
    assert pdf_cfg.reporting._target_ == "PDFReporter"
    assert pdf_cfg.reporting.multirun_report is True
    assert pdf_cfg.reporting.show_original_per_explainer is False
    assert pdf_cfg.reporting.show_redundant_robustness_panels is False
    assert pdf_cfg.hydra.callbacks.reporting_sweep._target_.endswith("ReportingSweepCallback")

    html_cfg = _compose_raitap_config(["reporting=html"])
    assert html_cfg.reporting._target_ == "HTMLReporter"
    assert html_cfg.reporting.multirun_report is True

    opt_out_cfg = _compose_raitap_config(["reporting=pdf", "reporting.multirun_report=false"])
    assert opt_out_cfg.reporting._target_ == "PDFReporter"
    assert opt_out_cfg.reporting.multirun_report is False

    originals_cfg = _compose_raitap_config(
        ["reporting=pdf", "reporting.show_original_per_explainer=true"]
    )
    assert originals_cfg.reporting.show_original_per_explainer is True

    redundant_robustness_cfg = _compose_raitap_config(
        ["reporting=pdf", "reporting.show_redundant_robustness_panels=true"]
    )
    assert redundant_robustness_cfg.reporting.show_redundant_robustness_panels is True


def test_reporting_sweep_callback_skips_when_multirun_report_disabled(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    sweep_dir = tmp_path / "multirun"
    sweep_dir.mkdir()
    _write_child_manifest(sweep_dir / "0", heading="Metrics A")
    create_report = SimpleNamespace(called=False)

    def _capture_report(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        create_report.called = True

    monkeypatch.setattr("raitap.reporting.hydra_callback.create_report", _capture_report)

    config = OmegaConf.create(
        {
            "experiment_name": "demo",
            "reporting": {
                "_target_": "PDFReporter",
                "filename": "report.pdf",
                "multirun_report": False,
            },
            "hydra": {"sweep": {"dir": str(sweep_dir)}},
        }
    )

    ReportingSweepCallback().on_multirun_end(config)

    assert create_report.called is False


def test_reporting_sweep_callback_skips_when_reporting_disabled(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    sweep_dir = tmp_path / "multirun"
    sweep_dir.mkdir()
    _write_child_manifest(sweep_dir / "0", heading="Metrics A")
    create_report = SimpleNamespace(called=False)

    def _capture_report(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        create_report.called = True

    monkeypatch.setattr("raitap.reporting.hydra_callback.create_report", _capture_report)

    config = OmegaConf.create(
        {
            "experiment_name": "demo",
            "reporting": {"_target_": None, "multirun_report": False},
            "hydra": {"sweep": {"dir": str(sweep_dir)}},
        }
    )

    ReportingSweepCallback().on_multirun_end(config)

    assert create_report.called is False


def _compose_raitap_config(overrides: list[str] | None = None) -> Any:
    with initialize_config_dir(version_base="1.3", config_dir=str(_configs_dir())):
        return compose(
            config_name="config",
            overrides=[] if overrides is None else overrides,
            return_hydra_config=True,
        )


def _configs_dir() -> Path:
    return (Path(__file__).resolve().parents[2] / "configs").resolve()


def _write_child_manifest(
    child_dir: Path,
    *,
    heading: str,
    table_rows: tuple[tuple[str, str], ...] = (("accuracy", "0.9000"),),
    include_cohort: bool = False,
    include_local: bool = False,
) -> None:
    report_dir = child_dir / "reports"
    report_dir.mkdir(parents=True)
    asset = _write_test_image(report_dir / "_assets" / "child.png")
    sections = [
        ReportSection.from_groups(
            "Metrics",
            [
                ReportGroup(
                    heading=heading,
                    images=(asset,),
                    table_rows=table_rows,
                    metadata={"role": "metrics"},
                )
            ],
        )
    ]
    if include_local:
        if include_cohort:
            cohort_asset = _write_test_image(report_dir / "_assets" / "cohort.png")
            sections.append(
                ReportSection.from_groups(
                    "Cohort Explanations",
                    [
                        ReportGroup(
                            heading=f"Cohort {heading}",
                            images=(cohort_asset,),
                            metadata={"role": "cohort"},
                        )
                    ],
                )
            )
        local_asset = _write_test_image(report_dir / "_assets" / "local.png")
        sections.append(
            ReportSection.from_groups(
                "Local Explanations",
                [
                    ReportGroup(
                        heading=f"Local {heading}",
                        images=(local_asset,),
                        metadata={"role": "local"},
                    )
                ],
            )
        )
    manifest = ReportManifest(
        kind="run",
        sections=tuple(sections),
        metadata={"experiment_name": child_dir.name},
    )
    manifest.write(report_dir / "report_manifest.json", report_dir=report_dir)
    hydra_dir = child_dir / ".hydra"
    hydra_dir.mkdir(exist_ok=True)
    (hydra_dir / "overrides.yaml").write_text("- transparency=demo\n", encoding="utf-8")


def _write_test_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.imshow([[0.0, 1.0], [1.0, 0.0]], cmap="viridis")
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", dpi=80)
    plt.close(fig)
    return path
