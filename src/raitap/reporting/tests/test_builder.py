from __future__ import annotations

import zipfile
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
from raitap.pipeline.outputs import ForwardOutput, PredictionSummary, RunOutputs
from raitap.reporting.builder import (
    BuiltReport,
    build_merged_report,
    build_report,
)
from raitap.reporting.factory import create_report
from raitap.reporting.hydra_callback import ReportingSweepCallback
from raitap.reporting.manifest import ReportManifest
from raitap.reporting.sample_selection import ReportSampleSelectionEntry
from raitap.reporting.sections import ReportGroup, ReportSection
from raitap.reporting.staging import _copy_asset, _strip_report_figure_titles
from raitap.robustness.contracts import (
    AssessmentKind,
    Objective,
    PerturbationBudget,
    PerturbationNorm,
    ReportFigureScope,
    RobustnessSemantics,
    RobustnessVerdict,
    RobustnessVisualisationContext,
    ThreatModel,
)
from raitap.robustness.report import (
    RobustnessPhaseResult,
    _canonical_facet_owners,
    _render_kwargs_for_robustness_visualiser,
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
from raitap.transparency.contracts import (
    DetectionBox,
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
from raitap.transparency.report import (
    TransparencyPhaseResult,
    _detection_box_heading,
    _overlay_detection_boxes,
    _overlay_legend_line,
)
from raitap.transparency.results import ConfiguredVisualiser, ExplanationResult, VisualisationResult
from raitap.transparency.visualisers import InputThumbnailVisualiser
from raitap.transparency.visualisers.base_visualiser import BaseVisualiser
from raitap.types import TaskKind


def _fo(tensor: torch.Tensor) -> ForwardOutput:
    return ForwardOutput(
        task_kind=TaskKind.classification,
        batch_size=int(tensor.shape[0]) if tensor.ndim > 0 else 0,
        payload=tensor,
    )


if TYPE_CHECKING:
    from matplotlib.figure import Figure


class _MetricsStub:
    """Stand-in MetricsEvaluation: a Trackable + Reportable metrics phase result."""

    report_order = 10

    def __init__(self, image_path: Path) -> None:
        self._image_path = image_path

    def to_report_group(self) -> ReportGroup:
        return ReportGroup(
            heading="Performance Metrics",
            images=(self._image_path,),
            table_rows=(("accuracy", "0.9000"),),
        )

    def log(self, tracker: Any, **kwargs: Any) -> None:
        del tracker, kwargs

    def report_sections(self, ctx: Any) -> tuple[ReportSection, ...]:
        source_group = self.to_report_group()
        staged = tuple(
            _copy_asset(p, assets_dir=ctx.assets_dir, target_name=f"metrics_{i}{p.suffix}")
            for i, p in enumerate(source_group.images)
        )
        group = ReportGroup(
            heading=source_group.heading,
            images=staged,
            table_rows=source_group.table_rows,
            metadata={"role": "metrics"},
        )
        return (
            ReportSection.from_groups("Metrics", [group], metadata={"section_role": "metrics"}),
        )


def _run_outputs(
    *,
    forward_output: ForwardOutput,
    explanations: list[Any] | None = None,
    visualisations: list[Any] | None = None,
    metrics: Any | None = None,
    robustness_results: list[Any] | None = None,
    robustness_visualisations: list[Any] | None = None,
    sample_ids: list[str] | None = None,
    targets: Any | None = None,
    prediction_summaries: tuple[PredictionSummary, ...] = (),
) -> RunOutputs:
    """Build a keyed-``RunOutputs`` from the legacy flat fields (test helper)."""
    phase_results: dict[str, Any] = {}
    if metrics is not None:
        phase_results["metrics"] = metrics
    if explanations or visualisations:
        # Visualisations now live on their owning result (issue #243); attach each
        # to its back-referenced explanation so the derived phase view flattens them.
        for visualisation in visualisations or []:
            visualisation.explanation.visualisations.append(visualisation)
        phase_results["transparency"] = TransparencyPhaseResult(
            explanations=list(explanations or []),
        )
    if robustness_results or robustness_visualisations:
        for visualisation in robustness_visualisations or []:
            visualisation.result.visualisations.append(visualisation)
        phase_results["robustness"] = RobustnessPhaseResult(
            results=list(robustness_results or []),
        )
    return RunOutputs(
        forward_output=forward_output,
        phase_results=phase_results,
        sample_ids=sample_ids,
        targets=targets,
        prediction_summaries=prediction_summaries,
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


class _ErroringRobustnessVisualiser(BaseRobustnessVisualiser):
    def visualise(
        self,
        result: RobustnessResult,
        *,
        context: RobustnessVisualisationContext,
        **kwargs: Any,
    ) -> Figure:
        del result, context, kwargs
        raise ValueError("builder visualiser failed")


class _AssessorScopeSamplingVisualiser(BaseRobustnessVisualiser):
    supported_assessment_kinds = frozenset({AssessmentKind.STATISTICAL_SAMPLING})
    report_figure_scope = ReportFigureScope.ASSESSOR

    def visualise(
        self,
        result: RobustnessResult,
        *,
        context: RobustnessVisualisationContext,
        **kwargs: Any,
    ) -> Figure:
        del result, context, kwargs
        fig, ax = plt.subplots()
        ax.bar([0, 1], [1.0, 0.5])
        return fig


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
        assessment_kind=AssessmentKind.EMPIRICAL_ATTACK,
        threat_model=ThreatModel.WHITE_BOX,
        objective=Objective.UNTARGETED,
        families=frozenset({"gradient_sign"}),
        perturbation=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.03),
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
        verdicts=encode_verdicts([RobustnessVerdict.ATTACK_SUCCEEDED] * batch_size),
        metrics=RobustnessMetrics(
            clean_accuracy=1.0,
            adversarial_accuracy=0.0,
            attack_success_rate=1.0,
        ),
        run_dir=tmp_path / "robustness" / assessor_name,
        experiment_name="robustness",
        adapter_target="t",
        algorithm="FGSM",
        name=assessor_name,
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
        adapter_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((3, 1, 4, 4)),
        name="captum_ig",
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

    outputs = _run_outputs(
        explanations=[explanation],
        visualisations=[native_global],
        metrics=_MetricsStub(metrics_image),  # type: ignore[arg-type]
        forward_output=_fo(torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.95, 0.05]])),
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
        adapter_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((3, 1, 4, 4)),
        name="captum_ig",
        visualisers=[ConfiguredVisualiser(visualiser=_LocalImageVisualiser())],
    )
    outputs = _run_outputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.95, 0.05]])),
        prediction_summaries=(
            PredictionSummary(sample_index=0, predicted_class=1, confidence=0.9, correct=None),
            PredictionSummary(sample_index=1, predicted_class=0, confidence=0.8, correct=None),
            PredictionSummary(sample_index=2, predicted_class=0, confidence=0.95, correct=None),
        ),
    )

    report = build_report(config, outputs)

    assert [section.title for section in report.sections] == ["Local Explanations"]


def test_build_report_places_aggregated_visualisations_between_global_and_local(
    tmp_path: Path,
) -> None:
    config = AppConfig(experiment_name="aggregated")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    metrics_image = _write_test_image(tmp_path / "metrics.png")
    explanation = ExplanationResult(
        attributions=torch.rand(2, 1, 4, 4),
        inputs=torch.rand(2, 1, 4, 4),
        run_dir=tmp_path / "transparency" / "exp",
        experiment_name="aggregated",
        adapter_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((2, 1, 4, 4)),
        name="captum_ig",
        visualisers=[ConfiguredVisualiser(visualiser=_LocalImageVisualiser())],
    )
    native_aggregated_path = _write_test_image(tmp_path / "native_aggregated.png")
    native_aggregated = VisualisationResult(
        explanation=explanation,
        figure=plt.figure(),
        visualiser_name="Aggregated_0",
        visualiser_target="test.Aggregated_0",
        output_path=native_aggregated_path,
        scope=ExplanationScope.AGGREGATED,
    )
    outputs = _run_outputs(
        explanations=[explanation],
        visualisations=[native_aggregated],
        metrics=_MetricsStub(metrics_image),  # type: ignore[arg-type]
        forward_output=_fo(torch.tensor([[0.1, 0.9], [0.8, 0.2]])),
        prediction_summaries=(
            PredictionSummary(sample_index=0, predicted_class=1, confidence=0.9),
            PredictionSummary(sample_index=1, predicted_class=0, confidence=0.8),
        ),
    )

    report = build_report(config, outputs)

    assert [section.title for section in report.sections] == [
        "Metrics",
        "Aggregated Explanations",
        "Local Explanations",
    ]
    assert report.sections[1].groups[0].metadata["role"] == "aggregated"


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
        adapter_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((3, 1, 4, 4)),
        name="captum_ig",
        kwargs={"sample_names": ["a", "b", "c"], "show_sample_names": True},
        visualisers=[ConfiguredVisualiser(visualiser=visualiser)],
    )
    outputs = _run_outputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.95, 0.05]])),
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
        "raitap.transparency.report.InputThumbnailVisualiser.visualise",
        _titled_thumbnail,
    )

    explanation = ExplanationResult(
        attributions=torch.rand(1, 1, 4, 4),
        inputs=torch.rand(1, 1, 4, 4),
        run_dir=tmp_path / "transparency" / "exp",
        experiment_name="compact_thumbnail_titles",
        adapter_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((1, 1, 4, 4)),
        name="captum_ig",
        visualisers=[ConfiguredVisualiser(visualiser=_LocalImageVisualiser())],
    )
    outputs = _run_outputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.tensor([[0.1, 0.9]])),
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
        adapter_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((1, 1, 4, 4)),
        name="captum_ig",
        visualisers=[
            ConfiguredVisualiser(visualiser=compact_visualiser),
            ConfiguredVisualiser(visualiser=masked_visualiser),
        ],
    )
    outputs = _run_outputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.tensor([[0.1, 0.9]])),
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
        adapter_target="raitap.transparency.CaptumExplainer",
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
        name="gradcam_localisation",
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
    outputs = _run_outputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.tensor([[0.1, 0.9], [0.8, 0.2]])),
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
        adapter_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((2, 1, 4, 4)),
        name="captum_ig",
        kwargs={"sample_names": ["legacy-a", "legacy-b"], "show_sample_names": True},
        visualisers=[ConfiguredVisualiser(visualiser=visualiser)],
    )
    outputs = _run_outputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.tensor([[0.1, 0.9], [0.8, 0.2]])),
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
        adapter_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((1, 1, 4, 4)),
        name="image_exp",
        visualisers=[ConfiguredVisualiser(visualiser=image_visualiser)],
    )
    tabular_explanation = ExplanationResult(
        attributions=torch.rand(1, 4),
        inputs=torch.rand(1, 4),
        run_dir=tmp_path / "transparency" / "tabular",
        experiment_name="thumbnail_fallback",
        adapter_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_tabular_semantics((1, 4)),
        name="tabular_exp",
        visualisers=[ConfiguredVisualiser(visualiser=tabular_visualiser)],
    )
    outputs = _run_outputs(
        explanations=[tabular_explanation, image_explanation],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.tensor([[0.1, 0.9]])),
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


def test_build_report_thumbnail_falls_back_to_later_explanation_after_runtime_error(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    config = AppConfig(experiment_name="thumbnail_multi_fallback")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    original_visualise = InputThumbnailVisualiser.visualise
    call_count = {"n": 0}

    def _fail_first_then_delegate(self: Any, *args: Any, **kwargs: Any) -> Any:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("first explanation thumbnail failed")
        return original_visualise(self, *args, **kwargs)

    monkeypatch.setattr(
        "raitap.transparency.report.InputThumbnailVisualiser.visualise",
        _fail_first_then_delegate,
    )

    first_explanation = ExplanationResult(
        attributions=torch.rand(1, 1, 4, 4),
        inputs=torch.rand(1, 1, 4, 4),
        run_dir=tmp_path / "transparency" / "first",
        experiment_name="thumbnail_multi_fallback",
        adapter_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((1, 1, 4, 4)),
        name="first_exp",
        visualisers=[ConfiguredVisualiser(visualiser=_EmbeddedOriginalVisualiser())],
    )
    second_explanation = ExplanationResult(
        attributions=torch.rand(1, 1, 4, 4),
        inputs=torch.rand(1, 1, 4, 4),
        run_dir=tmp_path / "transparency" / "second",
        experiment_name="thumbnail_multi_fallback",
        adapter_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((1, 1, 4, 4)),
        name="second_exp",
        visualisers=[ConfiguredVisualiser(visualiser=_EmbeddedOriginalVisualiser())],
    )
    outputs = _run_outputs(
        explanations=[first_explanation, second_explanation],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.tensor([[0.1, 0.9]])),
        prediction_summaries=(
            PredictionSummary(sample_index=0, predicted_class=1, confidence=0.9),
        ),
    )

    report = build_report(config, outputs)

    local_groups = report.sections[0].groups
    sample_header_groups = [
        group for group in local_groups if group.metadata["role"] == "sample_header"
    ]
    assert len(sample_header_groups) == 1
    assert sample_header_groups[0].metadata["source_explainer_name"] == "second_exp"
    assert call_count["n"] == 2


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
        adapter_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_tabular_semantics((1, 4)),
        name="tabular_exp",
        visualisers=[ConfiguredVisualiser(visualiser=visualiser)],
    )
    outputs = _run_outputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.tensor([[0.1, 0.9]])),
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
        "raitap.transparency.report.InputThumbnailVisualiser.visualise",
        _raise_runtime_error,
    )

    visualiser = _EmbeddedOriginalVisualiser()
    explanation = ExplanationResult(
        attributions=torch.rand(1, 1, 4, 4),
        inputs=torch.rand(1, 1, 4, 4),
        run_dir=tmp_path / "transparency" / "exp",
        experiment_name="thumbnail_runtime_fallback",
        adapter_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((1, 1, 4, 4)),
        name="captum_ig",
        visualisers=[ConfiguredVisualiser(visualiser=visualiser)],
    )
    outputs = _run_outputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.tensor([[0.1, 0.9]])),
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
        "raitap.transparency.report.InputThumbnailVisualiser.visualise",
        _raise_type_error,
    )

    explanation = ExplanationResult(
        attributions=torch.rand(1, 1, 4, 4),
        inputs=torch.rand(1, 1, 4, 4),
        run_dir=tmp_path / "transparency" / "exp",
        experiment_name="thumbnail_programmer_error",
        adapter_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((1, 1, 4, 4)),
        name="captum_ig",
        visualisers=[ConfiguredVisualiser(visualiser=_EmbeddedOriginalVisualiser())],
    )
    outputs = _run_outputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.tensor([[0.1, 0.9]])),
        prediction_summaries=(
            PredictionSummary(sample_index=0, predicted_class=1, confidence=0.9),
        ),
    )

    with pytest.raises(TypeError, match="programmer error"):
        build_report(config, outputs)


def test_build_report_explicit_filenames_render_in_user_order(tmp_path: Path) -> None:
    config, outputs = _explicit_selection_case(tmp_path)
    config.reporting.sample_selection = [  # type: ignore[union-attr]
        "case_gamma.png",
        "case_alpha.png",
        "case_delta.png",
        "case_beta.png",
    ]

    report = build_report(config, outputs)

    local_groups = report.sections[0].groups
    assert [group.metadata["role"] for group in local_groups] == [
        "local_detail",
        "local_detail",
        "local_detail",
        "local_detail",
    ]
    assert [group.metadata["sample_index"] for group in local_groups] == [2, 0, 3, 1]
    assert [group.metadata["requested_sample"] for group in local_groups] == [
        "case_gamma.png",
        "case_alpha.png",
        "case_delta.png",
        "case_beta.png",
    ]
    assert all(group.metadata["selection_source"] == "user" for group in local_groups)
    assert all(group.heading.startswith("Detail - user_selected") for group in local_groups)
    assert [sample["sample_index"] for sample in report.manifest.metadata["selected_samples"]] == [
        2,
        0,
        3,
        1,
    ]


def test_build_report_html_explicit_selection_uses_compact_local_layout(
    tmp_path: Path,
) -> None:
    config, outputs = _explicit_selection_case(tmp_path)
    config.reporting = ReportingConfig(_target_="HTMLReporter", filename="report")
    config.reporting.sample_selection = [
        "case_gamma.png",
        "case_alpha.png",
        "case_delta.png",
        "case_beta.png",
    ]

    report = build_report(config, outputs)

    local_groups = report.sections[0].groups
    header_groups = [group for group in local_groups if group.metadata["role"] == "sample_header"]
    visualiser_groups = [
        group for group in local_groups if group.metadata["role"] == "local_visualiser"
    ]
    assert [group.metadata["sample_index"] for group in header_groups] == [2, 0, 3, 1]
    assert [group.metadata["requested_sample"] for group in header_groups] == [
        "case_gamma.png",
        "case_alpha.png",
        "case_delta.png",
        "case_beta.png",
    ]
    assert [group.metadata["sample_index"] for group in visualiser_groups] == [2, 0, 3, 1]
    assert all(group.images for group in header_groups)
    assert all(group.images for group in visualiser_groups)
    assert dict(header_groups[0].table_rows)["predicted_class"] == "1"
    assert dict(header_groups[0].table_rows)["confidence"] == "0.5200"


def test_build_report_explicit_filename_extension_normalisation(tmp_path: Path) -> None:
    config, outputs = _explicit_selection_case(tmp_path)
    config.reporting.sample_selection = ["case_beta.jpg"]  # type: ignore[union-attr]

    report = build_report(config, outputs)

    assert report.sections[0].groups[0].metadata["sample_index"] == 1


def test_build_report_explicit_integer_index_selection(tmp_path: Path) -> None:
    config, outputs = _explicit_selection_case(tmp_path)
    config.reporting.sample_selection = [3, 0]  # type: ignore[union-attr]

    report = build_report(config, outputs)

    assert [group.metadata["sample_index"] for group in report.sections[0].groups] == [3, 0]


def test_build_report_explicit_mixed_string_and_index_selection(tmp_path: Path) -> None:
    config, outputs = _explicit_selection_case(tmp_path)
    config.reporting.sample_selection = ["case_alpha.png", 2]  # type: ignore[union-attr]

    report = build_report(config, outputs)

    assert [group.metadata["sample_index"] for group in report.sections[0].groups] == [0, 2]


def test_build_report_explicit_invalid_filename_fails(tmp_path: Path) -> None:
    config, outputs = _explicit_selection_case(tmp_path)
    config.reporting.sample_selection = ["missing.png"]  # type: ignore[union-attr]

    with pytest.raises(ValueError, match=r"missing[.]png"):
        build_report(config, outputs)


def test_build_report_explicit_ambiguous_filename_fails(tmp_path: Path) -> None:
    config, outputs = _explicit_selection_case(
        tmp_path,
        sample_ids=["set_a/case_alpha.png", "set_b/case_alpha.png"],
    )
    config.reporting.sample_selection = ["case_alpha.png"]  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="ambiguous"):
        build_report(config, outputs)


def test_build_report_explicit_out_of_range_index_fails(tmp_path: Path) -> None:
    config, outputs = _explicit_selection_case(tmp_path)
    config.reporting.sample_selection = [4]  # type: ignore[union-attr]

    with pytest.raises(ValueError, match=r"valid range is 0\.\.3"):
        build_report(config, outputs)


def test_build_report_explicit_duplicate_selection_fails(tmp_path: Path) -> None:
    config, outputs = _explicit_selection_case(tmp_path)
    config.reporting.sample_selection = ["case_alpha.png", "case_alpha"]  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="Duplicate"):
        build_report(config, outputs)


@pytest.mark.parametrize(
    "sample_selection",
    ["case_alpha.png", 1, {"samples": ["case_alpha.png"]}],
)
def test_build_report_explicit_selection_rejects_non_list_config_shapes(
    tmp_path: Path,
    sample_selection: Any,
) -> None:
    config, outputs = _explicit_selection_case(tmp_path)
    config.reporting.sample_selection = sample_selection  # type: ignore[union-attr,assignment]

    with pytest.raises(ValueError, match=r"reporting[.]sample_selection must be a list"):
        build_report(config, outputs)


def test_build_report_explicit_selection_rejects_empty_list(tmp_path: Path) -> None:
    config, outputs = _explicit_selection_case(tmp_path)
    config.reporting.sample_selection = []  # type: ignore[union-attr]

    with pytest.raises(ValueError, match=r"must contain at least one sample"):
        build_report(config, outputs)


@pytest.mark.parametrize("sample_selection", [[True], [{"sample": "case_alpha.png"}], [[0]]])
def test_build_report_explicit_selection_rejects_unsupported_entries(
    tmp_path: Path,
    sample_selection: Any,
) -> None:
    config, outputs = _explicit_selection_case(tmp_path)
    config.reporting.sample_selection = sample_selection  # type: ignore[union-attr,assignment]

    with pytest.raises(ValueError, match="Unsupported report sample selection entry"):
        build_report(config, outputs)


def test_report_sample_selection_entry_type_documents_supported_values() -> None:
    assert ReportSampleSelectionEntry == int | str


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
        adapter_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((2, 1, 4, 4)),
        visualisers=[ConfiguredVisualiser(visualiser=_GlobalOnlyVisualiser())],
    )
    outputs = _run_outputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.tensor([[0.1, 0.9], [0.8, 0.2]])),
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


def test_robustness_render_kwargs_skip_if_all_declared_facets_would_be_omitted() -> None:
    visualiser = ImagePairVisualiser(max_samples=1)

    assert (
        _render_kwargs_for_robustness_visualiser(
            visualiser,
            owners={"clean_input": 1, "perturbation_map": 2},
            visualiser_index=0,
            omit_redundant=True,
        )
        is None
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
    outputs = _run_outputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.zeros(1, 2)),
        robustness_results=[result],
    )

    report = build_report(config, outputs)

    robustness_group = report.sections[0].groups[0]
    assert robustness_group.metadata["role"] == "robustness"
    assert len(robustness_group.images) == 1
    pair_image = plt.imread(robustness_group.images[0])
    width_ratio = pair_image.shape[1] / pair_image.shape[0]
    assert width_ratio > 2.0
    assert dict(robustness_group.table_rows)["attack_success_rate"] == "1.0000"
    assert robustness_group.images[0].name.startswith(
        "robustness_0_fgsm_sample_0_ImagePairVisualiser_0"
    )
    assert "PerturbationHeatmapVisualiser" not in robustness_group.images[0].name


def test_build_report_compact_robustness_skips_redundant_single_facet_visualiser(
    tmp_path: Path,
) -> None:
    config = AppConfig(experiment_name="robustness_compact_skip")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    owner = _PerturbationRecordingVisualiser()
    redundant = _PerturbationRecordingVisualiser()
    result = _make_robustness_result(
        tmp_path,
        visualisers=[
            ConfiguredRobustnessVisualiser(visualiser=owner),
            ConfiguredRobustnessVisualiser(visualiser=redundant),
        ],
    )
    outputs = _run_outputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.zeros(1, 2)),
        robustness_results=[result],
    )

    report = build_report(config, outputs)

    robustness_group = report.sections[0].groups[0]
    assert len(robustness_group.images) == 1
    assert owner.calls == [{}]
    assert redundant.calls == []
    assert robustness_group.images[0].name.startswith(
        "robustness_0_fgsm_sample_0__PerturbationRecordingVisualiser_0"
    )


def test_build_report_compact_robustness_propagates_visualiser_errors(
    tmp_path: Path,
) -> None:
    config = AppConfig(experiment_name="robustness_compact_error")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    result = _make_robustness_result(
        tmp_path,
        visualisers=[
            ConfiguredRobustnessVisualiser(visualiser=_ErroringRobustnessVisualiser()),
        ],
    )
    outputs = _run_outputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.zeros(1, 2)),
        robustness_results=[result],
    )

    with pytest.raises(ValueError, match="builder visualiser failed"):
        build_report(config, outputs)


def test_build_report_compact_robustness_renders_selected_samples_per_assessor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    outputs = _run_outputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.zeros(20, 2)),
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
    stripped_figures: list[tuple[str, tuple[str, ...]]] = []
    original_strip = _strip_report_figure_titles

    def _record_stripped_titles(figure: Figure) -> None:
        original_strip(figure)
        suptitle_artist = getattr(figure, "_suptitle", None)
        suptitle = suptitle_artist.get_text() if suptitle_artist is not None else ""
        stripped_figures.append((suptitle, tuple(ax.get_title() for ax in figure.axes)))

    monkeypatch.setattr(
        "raitap.robustness.report._strip_report_figure_titles", _record_stripped_titles
    )

    report = build_report(config, outputs)

    robustness_group = report.sections[0].groups[0]
    assert robustness_group.metadata["role"] == "robustness"
    assert robustness_group.metadata["sample_indices"] == (3, 8, 19)
    assert len(report.sections[0].groups) == 1
    assert len(robustness_group.images) == 3
    image_names = [image.name for image in robustness_group.images]
    assert all("_sample_" in name for name in image_names)
    assert [name.split("_sample_", 1)[1].split("_", 1)[0] for name in image_names] == [
        "3",
        "8",
        "19",
    ]
    assert all("ImagePairVisualiser_0" in name for name in image_names)
    assert all("PerturbationHeatmapVisualiser" not in name for name in image_names)
    assert any("sample_3_ImagePairVisualiser_0" in name for name in image_names)
    assert any("sample_8_ImagePairVisualiser_0" in name for name in image_names)
    assert any("sample_19_ImagePairVisualiser_0" in name for name in image_names)
    assert len(stripped_figures) == 3
    assert all(suptitle == "" for suptitle, _titles in stripped_figures)
    assert all(all(title == "" for title in titles) for _suptitle, titles in stripped_figures)


def test_build_report_robustness_single_pair_keeps_all_panels(tmp_path: Path) -> None:
    config = AppConfig(experiment_name="robustness_single_pair")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    result = _make_robustness_result(
        tmp_path,
        visualisers=[ConfiguredRobustnessVisualiser(visualiser=ImagePairVisualiser(max_samples=1))],
    )
    outputs = _run_outputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.zeros(1, 2)),
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
    outputs = _run_outputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.zeros(1, 2)),
        robustness_results=[result],
        robustness_visualisations=[existing_visualisation],
    )

    report = build_report(config, outputs)

    assert len(report.sections[0].groups[0].images) == 1
    assert recording.calls == []


def test_build_report_robustness_redundant_single_facet_without_kwarg_support_is_skipped(
    tmp_path: Path,
) -> None:
    config = AppConfig(experiment_name="robustness_redundant_strict_visualiser")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    result = _make_robustness_result(
        tmp_path,
        visualisers=[
            ConfiguredRobustnessVisualiser(visualiser=PerturbationHeatmapVisualiser(max_samples=1)),
            ConfiguredRobustnessVisualiser(visualiser=_StrictPerturbationVisualiser()),
        ],
    )
    outputs = _run_outputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.zeros(1, 2)),
        robustness_results=[result],
    )

    report = build_report(config, outputs)

    assert len(report.sections[0].groups[0].images) == 1


def test_report_manifest_round_trip_preserves_relative_images(tmp_path: Path) -> None:
    config = AppConfig(experiment_name="demo")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    explanation = ExplanationResult(
        attributions=torch.rand(2, 1, 4, 4),
        inputs=torch.rand(2, 1, 4, 4),
        run_dir=tmp_path / "transparency" / "exp",
        experiment_name="demo",
        adapter_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((2, 1, 4, 4)),
        name="captum_ig",
        visualisers=[ConfiguredVisualiser(visualiser=_LocalImageVisualiser())],
    )
    outputs = _run_outputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.tensor([[0.1, 0.9], [0.8, 0.2]])),
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


def test_create_report_writes_html_archive_with_manifest_and_assets(
    tmp_path: Path, monkeypatch: Any
) -> None:
    config = AppConfig(experiment_name="demo")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="HTMLReporter", filename="report.html")
    built_report = ReportSection.from_groups(
        "Metrics",
        [ReportGroup(heading="Performance Metrics", table_rows=(("accuracy", "0.9"),))],
    )
    report = ReportManifest(kind="run", sections=(built_report,), filename="report.html")
    report_dir = tmp_path / "builder-reports"
    asset_dir = report_dir / "_assets"
    asset_dir.mkdir(parents=True)
    (asset_dir / "example.png").write_bytes(b"png")
    built = BuiltReport(report_dir=report_dir, sections=(built_report,), manifest=report)

    class _HTMLReporterStub:
        def __init__(self, _config: Any) -> None:
            pass

        def generate(self, sections: Any, *, report_dir: Path | None = None) -> Path:
            del sections
            assert report_dir is not None
            output_path = report_dir / "report.html"
            output_path.write_text("<html></html>", encoding="utf-8")
            return output_path

    monkeypatch.setattr("raitap.reporting.factory.instantiate", lambda _cfg: _HTMLReporterStub)

    generated = create_report(config, built)

    archive_path = generated.report_path.with_suffix(".zip")
    assert archive_path.exists()
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == [
            "report.html",
            "report_manifest.json",
            "_assets/example.png",
        ]


def test_create_report_does_not_archive_pdf_report(tmp_path: Path, monkeypatch: Any) -> None:
    config = AppConfig(experiment_name="demo")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")
    built_report = ReportSection.from_groups(
        "Metrics",
        [ReportGroup(heading="Performance Metrics", table_rows=(("accuracy", "0.9"),))],
    )
    report = ReportManifest(kind="run", sections=(built_report,), filename="report.pdf")
    built = BuiltReport(
        report_dir=tmp_path / "builder-reports",
        sections=(built_report,),
        manifest=report,
    )

    class _PDFReporterStub:
        def __init__(self, _config: Any) -> None:
            pass

        def generate(self, sections: Any, *, report_dir: Path | None = None) -> Path:
            del sections
            assert report_dir is not None
            output_path = report_dir / "report.pdf"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"%PDF-1.4\n")
            return output_path

    monkeypatch.setattr("raitap.reporting.factory.instantiate", lambda _cfg: _PDFReporterStub)

    generated = create_report(config, built)

    assert not generated.report_path.with_suffix(".zip").exists()


@pytest.mark.parametrize(
    ("configured_filename", "expected_html_name", "expected_pdf_name"),
    [
        ("custom_report", "custom_report.html", "custom_report.pdf"),
        ("custom_report.pdf", "custom_report.html", "custom_report.pdf"),
        ("custom_report.html", "custom_report.html", "custom_report.pdf"),
    ],
)
def test_build_report_manifest_filename_matches_selected_reporter(
    tmp_path: Path,
    configured_filename: str,
    expected_html_name: str,
    expected_pdf_name: str,
) -> None:
    outputs = _run_outputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.empty(0)),
    )

    html_config = AppConfig(experiment_name="demo")
    set_output_root(html_config, tmp_path / "html")
    html_config.reporting = ReportingConfig(
        _target_="raitap.reporting.HTMLReporter",
        filename=configured_filename,
    )
    html_report = build_report(html_config, outputs)

    assert html_report.manifest.filename == expected_html_name

    pdf_config = AppConfig(experiment_name="demo")
    set_output_root(pdf_config, tmp_path / "pdf")
    pdf_config.reporting = ReportingConfig(
        _target_="raitap.reporting.PDFReporter",
        filename=configured_filename,
    )
    pdf_report = build_report(pdf_config, outputs)

    assert pdf_report.manifest.filename == expected_pdf_name


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


def test_build_merged_report_preserves_present_section_order_with_aggregated(
    tmp_path: Path,
) -> None:
    sweep_dir = tmp_path / "multirun"
    sweep_dir.mkdir()
    _write_child_manifest(
        sweep_dir / "0",
        heading="Metrics A",
        include_aggregated=True,
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
        "Aggregated Explanations",
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
    """Bundled reporting presets resolve the right ``_target_`` and wire the
    multirun-aggregation Hydra callback.

    The presets ship as minimal ``_target_``-only stubs (plus the
    ``reporting_sweep`` callback for non-disabled presets). All other
    ReportingConfig fields come from the user's config or CLI overrides;
    this test only verifies what's actually shipped + the callback wiring.
    """
    pdf_cfg = _compose_raitap_config(["+reporting=pdf"])
    assert pdf_cfg.reporting._target_ == "PDFReporter"
    assert pdf_cfg.hydra.callbacks.reporting_sweep._target_.endswith("ReportingSweepCallback")

    html_cfg = _compose_raitap_config(["+reporting=html"])
    assert html_cfg.reporting._target_ == "HTMLReporter"
    assert html_cfg.hydra.callbacks.reporting_sweep._target_.endswith("ReportingSweepCallback")

    disabled_cfg = _compose_raitap_config(["+reporting=disabled"])
    assert disabled_cfg.reporting._target_ is None
    assert disabled_cfg.reporting.multirun_report is False


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
            config_name="demo",
            overrides=[] if overrides is None else overrides,
            return_hydra_config=True,
        )


def _configs_dir() -> Path:
    return (Path(__file__).resolve().parents[2] / "configs").resolve()


def _explicit_selection_case(
    tmp_path: Path,
    *,
    sample_ids: list[str] | None = None,
) -> tuple[AppConfig, RunOutputs]:
    ids = sample_ids or [
        "case_alpha.png",
        "case_beta.png",
        "case_gamma.png",
        "case_delta.png",
    ]
    config = AppConfig(experiment_name="explicit_selection")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")
    explanation = ExplanationResult(
        attributions=torch.rand(len(ids), 1, 4, 4),
        inputs=torch.rand(len(ids), 1, 4, 4),
        run_dir=tmp_path / "transparency" / "exp",
        experiment_name="explicit_selection",
        adapter_target="t",
        algorithm="IntegratedGradients",
        semantics=_local_image_semantics((len(ids), 1, 4, 4)),
        name="captum_ig",
        kwargs={"sample_names": ids, "show_sample_names": True},
        visualisers=[ConfiguredVisualiser(visualiser=_LocalImageVisualiser())],
    )
    outputs = _run_outputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.rand(len(ids), 2)),
        sample_ids=ids,
        prediction_summaries=tuple(
            PredictionSummary(
                sample_index=index,
                sample_id=sample_id,
                predicted_class=1,
                confidence=0.5 + index * 0.01,
            )
            for index, sample_id in enumerate(ids)
        ),
    )
    return config, outputs


def _write_child_manifest(
    child_dir: Path,
    *,
    heading: str,
    table_rows: tuple[tuple[str, str], ...] = (("accuracy", "0.9000"),),
    include_aggregated: bool = False,
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
        if include_aggregated:
            aggregated_asset = _write_test_image(report_dir / "_assets" / "aggregated.png")
            sections.append(
                ReportSection.from_groups(
                    "Aggregated Explanations",
                    [
                        ReportGroup(
                            heading=f"Aggregated {heading}",
                            images=(aggregated_asset,),
                            metadata={"role": "aggregated"},
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


def test_build_report_sampling_result_renders_without_error(tmp_path: Path) -> None:
    from raitap.robustness.contracts import PerturbationDistribution

    config = AppConfig(experiment_name="sampling_test")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    semantics = RobustnessSemantics(
        assessment_kind=AssessmentKind.STATISTICAL_SAMPLING,
        threat_model=ThreatModel.NOT_APPLICABLE,
        objective=Objective.UNTARGETED,
        families=frozenset({"noise"}),
        perturbation=PerturbationDistribution(corruption_name="fog", severity=3),
    )
    result = RobustnessResult(
        clean_inputs=torch.rand(2, 3, 4, 4),
        targets=torch.tensor([0, 1]),
        clean_predictions=torch.tensor([0, 1]),
        verdicts=encode_verdicts([RobustnessVerdict.ATTACK_SUCCEEDED] * 2),
        metrics=RobustnessMetrics(
            clean_accuracy=1.0,
            corrupted_accuracy=0.5,
            n_samples=2,
            n_correct=1,
        ),
        run_dir=tmp_path / "robustness" / "fog",
        experiment_name="sampling_test",
        adapter_target="t",
        algorithm="fog",
        name="fog",
        semantics=semantics,
        visualisers=[],
    )
    outputs = _run_outputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.zeros(2, 2)),
        robustness_results=[result],
    )

    report = build_report(config, outputs)

    assert len(report.sections) == 1
    robustness_section = report.sections[0]
    assert robustness_section.title == "Robustness"
    group = robustness_section.groups[0]
    row_dict = dict(group.table_rows)
    assert "Average-case" in group.heading
    assert row_dict.get("corruption_name") == "fog"
    assert row_dict.get("severity") == "3"
    assert row_dict.get("case") == "average_case"
    assert row_dict.get("corrupted_accuracy") == "0.5000"


def test_build_report_assessor_scope_figure_recorded_in_metadata(tmp_path: Path) -> None:
    from raitap.robustness.contracts import PerturbationDistribution

    config = AppConfig(experiment_name="sampling_scope")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    semantics = RobustnessSemantics(
        assessment_kind=AssessmentKind.STATISTICAL_SAMPLING,
        threat_model=ThreatModel.NOT_APPLICABLE,
        objective=Objective.UNTARGETED,
        families=frozenset({"noise"}),
        perturbation=PerturbationDistribution(corruption_name="fog", severity=3),
    )
    result = RobustnessResult(
        clean_inputs=torch.rand(2, 3, 4, 4),
        targets=torch.tensor([0, 1]),
        clean_predictions=torch.tensor([0, 1]),
        verdicts=encode_verdicts([RobustnessVerdict.CORRECT_UNDER_PERTURBATION] * 2),
        metrics=RobustnessMetrics(clean_accuracy=1.0, corrupted_accuracy=0.5, n_samples=2),
        run_dir=tmp_path / "robustness" / "fog",
        experiment_name="sampling_scope",
        adapter_target="t",
        algorithm="fog",
        name="fog",
        semantics=semantics,
        visualisers=[
            ConfiguredRobustnessVisualiser(visualiser=_AssessorScopeSamplingVisualiser()),
        ],
    )
    outputs = _run_outputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.zeros(2, 2)),
        robustness_results=[result],
    )

    report = build_report(config, outputs)

    group = report.sections[0].groups[0]
    assert len(group.images) == 1
    staged_name = group.images[0].name
    # Assessor-level figures carry no per-sample token and are tagged in metadata.
    assert "_sample_" not in staged_name
    figure_scopes = group.metadata["figure_scopes"]
    assert isinstance(figure_scopes, dict)
    assert figure_scopes[staged_name] == ReportFigureScope.ASSESSOR.value


def _write_test_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.imshow([[0.0, 1.0], [1.0, 0.0]], cmap="viridis")
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", dpi=80)
    plt.close(fig)
    return path


def test_baseline_mode_label_humanises_known_tokens() -> None:
    from raitap.transparency.report import _baseline_mode_label

    assert _baseline_mode_label("configured") == "configured dataset"
    assert _baseline_mode_label("user_tensor") == "user-provided tensor"
    assert _baseline_mode_label("zero") == "all-zeros (method default)"
    assert _baseline_mode_label("input_batch") == "input batch (method default)"
    assert _baseline_mode_label("future_mode") == "future_mode"  # unknown -> passthrough


def test_transparency_table_rows_include_baseline_and_hide_opaque_kwarg(tmp_path: Path) -> None:
    from pathlib import Path as _Path

    from raitap.transparency.contracts import BaselineRecord
    from raitap.transparency.report import _transparency_table_rows

    record = BaselineRecord(
        kwarg_name="background_data",
        mode="configured",
        source="imagenet",
        n_samples=50,
        shape=(50, 3, 4, 4),
        dtype="torch.float32",
        sha256="secret-hash",
        image_path=_Path("baseline.png"),
    )
    explanation = ExplanationResult(
        attributions=torch.rand(1, 1, 4, 4),
        inputs=torch.rand(1, 1, 4, 4),
        run_dir=tmp_path / "exp",
        experiment_name="demo",
        adapter_target="t",
        algorithm="GradientExplainer",
        name="shap_grad",
        semantics=_local_image_semantics((1, 1, 4, 4)),
        # Small tensor (numel == 4) so it WOULD render as call.background_data
        # without suppression — proving the suppression branch is exercised.
        call_kwargs={"background_data": torch.zeros(4), "target": 0},
        visualisers=[ConfiguredVisualiser(visualiser=_LocalImageVisualiser())],
        baseline=record,
    )

    rows = dict(_transparency_table_rows(explanation, selected_samples=[], visualiser_index=0))

    # Mode token is humanised for the report; raw token stays in metadata.json.
    assert rows["baseline.mode"] == "configured dataset"
    assert rows["baseline.source"] == "imagenet"
    assert rows["baseline.n_samples"] == "50"
    assert "baseline.shape" in rows
    # sha256 never reaches the report.
    assert "secret-hash" not in str(rows)
    assert not any(k.startswith("baseline.sha2") for k in rows)
    # The opaque tensor kwarg is suppressed in favour of the labelled rows.
    assert "call.background_data" not in rows
    # Non-baseline kwargs still render.
    assert "call.target" in rows


def test_stage_baseline_image_copies_when_present(tmp_path: Path) -> None:
    from pathlib import Path as _Path
    from types import SimpleNamespace

    from raitap.transparency.contracts import BaselineRecord
    from raitap.transparency.report import _stage_baseline_image

    run_dir = tmp_path / "exp"
    run_dir.mkdir()
    _write_test_image(run_dir / "baseline.png")  # existing helper in this module
    record = BaselineRecord(
        kwarg_name="baselines",
        mode="zero",
        source=None,
        n_samples=None,
        shape=(1, 1, 4, 4),
        dtype="torch.float32",
        sha256="h",
        image_path=_Path("baseline.png"),
    )
    explanation = SimpleNamespace(baseline=record, run_dir=run_dir)

    out = _stage_baseline_image(explanation, assets_dir=tmp_path / "assets", stem="s0")
    assert out is not None
    assert out.exists()
    assert out.name == "baseline_s0.png"


def test_stage_baseline_image_none_when_no_baseline_or_missing_file(tmp_path: Path) -> None:
    from pathlib import Path as _Path
    from types import SimpleNamespace

    from raitap.transparency.contracts import BaselineRecord
    from raitap.transparency.report import _stage_baseline_image

    no_baseline = SimpleNamespace(baseline=None, run_dir=tmp_path)
    assert _stage_baseline_image(no_baseline, assets_dir=tmp_path / "a", stem="s") is None

    record = BaselineRecord(
        kwarg_name="baselines",
        mode="zero",
        source=None,
        n_samples=None,
        shape=(1, 1, 4, 4),
        dtype="torch.float32",
        sha256="h",
        image_path=_Path("baseline.png"),
    )
    missing = SimpleNamespace(baseline=record, run_dir=tmp_path / "missing")
    assert _stage_baseline_image(missing, assets_dir=tmp_path / "a", stem="s") is None

    # A baseline with no rendered image (e.g. tabular modality) stages nothing.
    no_image_record = BaselineRecord(
        kwarg_name="baselines",
        mode="zero",
        source=None,
        n_samples=None,
        shape=(1, 4),
        dtype="torch.float32",
        sha256="h",
        image_path=None,
    )
    no_image = SimpleNamespace(baseline=no_image_record, run_dir=tmp_path)
    assert _stage_baseline_image(no_image, assets_dir=tmp_path / "a", stem="s") is None


def test_build_report_attaches_baseline_image_once_per_explanation(tmp_path: Path) -> None:
    from pathlib import Path as _Path

    from raitap.transparency.contracts import BaselineRecord

    config = AppConfig(experiment_name="bl")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    run_dir = tmp_path / "transparency" / "exp"
    run_dir.mkdir(parents=True)
    _write_test_image(run_dir / "baseline.png")

    record = BaselineRecord(
        kwarg_name="baselines",
        mode="zero",
        source=None,
        n_samples=None,
        shape=(1, 1, 4, 4),
        dtype="torch.float32",
        sha256="h",
        image_path=_Path("baseline.png"),
    )
    explanation = ExplanationResult(
        attributions=torch.rand(1, 1, 4, 4),
        inputs=torch.rand(1, 1, 4, 4),
        run_dir=run_dir,
        experiment_name="bl",
        adapter_target="t",
        algorithm="IntegratedGradients",
        name="captum_ig",
        semantics=_local_image_semantics((1, 1, 4, 4)),
        visualisers=[
            ConfiguredVisualiser(visualiser=_LocalImageVisualiser()),
            ConfiguredVisualiser(visualiser=_LocalImageVisualiser()),
        ],
        baseline=record,
    )
    outputs = _run_outputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=_fo(torch.tensor([[0.1, 0.9]])),
        sample_ids=["a"],
        prediction_summaries=(
            PredictionSummary(
                sample_index=0,
                sample_id="a",
                predicted_class=1,
                target_class=0,
                confidence=0.9,
                correct=False,
            ),
        ),
    )

    report = build_report(config, outputs)

    local = next(s for s in report.sections if s.title == "Local Explanations")
    baseline_imgs = [
        p for group in local.groups for p in group.images if p.name.startswith("baseline_")
    ]
    local_vis_groups = [g for g in local.groups if g.metadata.get("role") == "local_visualiser"]
    # Two visualisers -> two local_visualiser groups, but the baseline renders once.
    assert len(local_vis_groups) == 2
    assert len(baseline_imgs) == 1


# ---------------------------------------------------------------------------
# _detection_box_heading unit tests (issue #233)
# ---------------------------------------------------------------------------


def test_detection_heading_matched_gt() -> None:
    box = DetectionBox(
        display_index=0,
        raw_index=2,
        xyxy=(0, 0, 1, 1),
        score=0.99,
        label_index=38,
        label_name="kite",
        ground_truth_evaluated=True,
        true_label_index=20,
        true_label_name="sheep",
        true_match_iou=0.71,
    )
    assert "pred: kite 0.99" in _detection_box_heading(box)
    assert "gt: sheep (IoU 0.71)" in _detection_box_heading(box)


def test_detection_heading_matched_gt_without_iou() -> None:
    # Defensive: true label set but no match IoU -> name shown, no "(IoU ...)".
    box = DetectionBox(
        display_index=0,
        raw_index=2,
        xyxy=(0, 0, 1, 1),
        score=0.99,
        label_index=38,
        label_name="kite",
        ground_truth_evaluated=True,
        true_label_index=20,
        true_label_name="sheep",
    )
    assert _detection_box_heading(box) == "pred: kite 0.99 | gt: sheep"


def test_detection_heading_no_match() -> None:
    box = DetectionBox(
        display_index=0,
        raw_index=2,
        xyxy=(0, 0, 1, 1),
        score=0.99,
        label_index=38,
        label_name="kite",
        ground_truth_evaluated=True,
    )
    assert "gt: no match" in _detection_box_heading(box)


def test_detection_heading_no_gt_unchanged() -> None:
    box = DetectionBox(
        display_index=0,
        raw_index=2,
        xyxy=(0, 0, 1, 1),
        score=0.99,
        label_index=38,
        label_name="kite",
    )
    h = _detection_box_heading(box)
    assert "gt:" not in h
    assert "kite, score=0.99" in h


def _det_box(
    *,
    display_index: int,
    label_name: str,
    score: float,
    ground_truth_evaluated: bool = False,
    true_label_name: str | None = None,
    true_label_index: int | None = None,
    true_match_iou: float | None = None,
) -> DetectionBox:
    return DetectionBox(
        display_index=display_index,
        raw_index=display_index,
        xyxy=(10.0, 10.0, 40.0, 40.0),
        score=score,
        label_index=1,
        label_name=label_name,
        ground_truth_evaluated=ground_truth_evaluated,
        true_label_name=true_label_name,
        true_label_index=true_label_index,
        true_match_iou=true_match_iou,
    )


def test_overlay_legend_line_covers_all_branches() -> None:
    matched = _det_box(
        display_index=0,
        label_name="kite",
        score=0.99,
        ground_truth_evaluated=True,
        true_label_name="sheep",
        true_label_index=3,
        true_match_iou=0.71,
    )
    assert _overlay_legend_line(matched) == "#0 kite (0.99) | gt: sheep (IoU 0.71)"
    no_match = _det_box(display_index=1, label_name="dog", score=0.92, ground_truth_evaluated=True)
    assert _overlay_legend_line(no_match) == "#1 dog (0.92) | gt: no match"
    no_gt = _det_box(display_index=2, label_name="boat", score=0.81)
    assert _overlay_legend_line(no_gt) == "#2 boat (0.81)"
    # Defensive: a true label without a match IoU shows the name, no "(IoU ...)".
    label_only = _det_box(
        display_index=3,
        label_name="cat",
        score=0.7,
        ground_truth_evaluated=True,
        true_label_name="cat",
        true_label_index=5,
    )
    assert _overlay_legend_line(label_only) == "#3 cat (0.70) | gt: cat"


class _DetExpl:
    def __init__(self, box: DetectionBox, sample_index: int) -> None:
        self.detection_box = box
        self.original_sample_index = sample_index


class _DetOutputs:
    def __init__(self, explanations: list[_DetExpl]) -> None:
        self.explanations = explanations


def test_overlay_draws_index_tags_and_legend_below_image() -> None:
    # Two clustered boxes on one sample: the image carries only compact #i tags
    # (no long labels that would collide), and one legend text holds both full
    # lines keyed by index.
    box0 = _det_box(
        display_index=0,
        label_name="kite",
        score=0.99,
        ground_truth_evaluated=True,
        true_label_name="kite",
        true_label_index=1,
        true_match_iou=0.83,
    )
    box1 = _det_box(display_index=1, label_name="dog", score=0.92, ground_truth_evaluated=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(torch.zeros(50, 50, 3).numpy())
    _overlay_detection_boxes(
        fig,
        outputs=_DetOutputs([_DetExpl(box0, 0), _DetExpl(box1, 0)]),  # type: ignore[arg-type]
        sample_index=0,
    )
    texts = [t.get_text() for t in ax.texts]
    # compact tags on the image
    assert "#0" in texts
    assert "#1" in texts
    # full detail in a single legend text, both lines, keyed by index
    legend = next(t for t in texts if "\n" in t)
    assert "#0 kite (0.99) | gt: kite (IoU 0.83)" in legend
    assert "#1 dog (0.92) | gt: no match" in legend
    # no long per-box label is stamped beside the boxes (only tags + legend)
    assert not any(t.startswith("#0 kite") for t in texts if "\n" not in t)
    plt.close(fig)


def test_overlay_dedups_boxes_repeated_across_explainers() -> None:
    """Regression for #241: each physical box is explained by every explainer, so
    ``outputs.explanations`` holds one result per (box x explainer). The overlay
    must draw each box ONCE — one rectangle, one ``#i`` tag, one legend line —
    not once per explainer, and the drawn set must match the legend 1:1.
    """
    from matplotlib.patches import Rectangle

    # Two physical boxes, each explained by three explainers => three distinct
    # DetectionBox instances per box, all sharing one display_index. Distinct
    # instances (not the same object) prove the dedup keys on display_index value,
    # not object identity.
    explanations: list[_DetExpl] = []
    for _ in range(3):  # IntegratedGradients / LayerGradCam / SHAP
        explanations.append(_DetExpl(_det_box(display_index=0, label_name="horse", score=0.85), 0))
        explanations.append(
            _DetExpl(_det_box(display_index=1, label_name="stop sign", score=0.73), 0)
        )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(torch.zeros(50, 50, 3).numpy())
    _overlay_detection_boxes(
        fig,
        outputs=_DetOutputs(explanations),  # type: ignore[arg-type]
        sample_index=0,
    )

    texts = [t.get_text() for t in ax.texts]
    legend = next(t for t in texts if "\n" in t)
    legend_lines = legend.split("\n")
    tags = [t for t in texts if "\n" not in t]  # compact #i image tags
    rects = [p for p in ax.patches if isinstance(p, Rectangle)]

    # Each physical box drawn once; legend reflects exactly the drawn set (1:1).
    assert len(rects) == len(legend_lines) == len(tags) == 2
    assert legend_lines == ["#0 horse (0.85)", "#1 stop sign (0.73)"]
    assert sorted(tags) == ["#0", "#1"]
    plt.close(fig)


def test_build_report_handles_reporting_block_without_sample_selection(
    tmp_path: Path,
) -> None:
    """Regression (sibling of #240): a struct-mode ``reporting:`` block that
    omits the optional ``sample_selection`` key must not crash report building.

    Like ``model.class_names`` in #240, ``build_report`` read
    ``reporting_cfg.sample_selection`` unconditionally. When ``config.reporting``
    arrives as a YAML-loaded struct-mode ``DictConfig`` (``object_type=dict``)
    that never declares the key, the attribute access raised
    ``ConfigAttributeError`` before ``resolve_report_sample_selection`` could
    apply its default. The read must be defensive, matching the ``getattr``
    guard the orchestrator already uses for the same key.
    """
    config = AppConfig(experiment_name="no-sample-selection")
    set_output_root(config, tmp_path)
    # Build a struct-mode reporting block carrying every ReportingConfig field
    # EXCEPT ``sample_selection`` — isolates the failure to the line-51 read.
    reporting_dict = OmegaConf.to_container(
        OmegaConf.structured(ReportingConfig(_target_="PDFReporter", filename="report.pdf")),
        resolve=True,
    )
    assert isinstance(reporting_dict, dict)
    reporting_dict.pop("sample_selection", None)
    reporting_block = OmegaConf.create(reporting_dict)
    OmegaConf.set_struct(reporting_block, True)
    config.reporting = reporting_block  # type: ignore[assignment]

    outputs = _run_outputs(
        forward_output=_fo(torch.tensor([[0.1, 0.9]])),
        sample_ids=["a"],
    )

    # Must not raise ConfigAttributeError on the optional-key read.
    report = build_report(config, outputs)
    assert report is not None
