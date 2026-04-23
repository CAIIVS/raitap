from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from raitap.configs import set_output_root
from raitap.configs.schema import AppConfig, ReportingConfig
from raitap.reporting.builder import build_report
from raitap.reporting.hydra_callback import ReportingSweepCallback
from raitap.reporting.manifest import ReportManifest
from raitap.reporting.sections import ReportGroup, ReportSection
from raitap.run.outputs import PredictionSummary, RunOutputs
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
    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        *,
        context: Any = None,
        **kwargs: Any,
    ) -> Figure:
        del attributions, inputs, kwargs
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow([[0.0, 1.0], [1.0, 0.0]], cmap="magma")
        if context is not None and context.sample_names:
            ax.set_title(str(context.sample_names[0]))
        ax.axis("off")
        return fig


class _GlobalVisualiser(BaseVisualiser):
    report_scope = "global"

    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        *,
        context: Any = None,
        **kwargs: Any,
    ) -> Figure:
        del attributions, inputs, context, kwargs
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1], [0, 1])
        ax.set_title("global")
        return fig


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
        report_scope="global",
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
    local_headings = [group.heading for group in report.sections[2].groups]
    assert local_headings[0].startswith("Overview - Explainer: captum_ig - wrong")
    assert "Detail - insecure" in local_headings[1]
    assert "Detail - high_confidence" in local_headings[2]
    assert all("Detail - wrong" not in heading for heading in local_headings)
    assert all(
        path.parent.name == "_assets"
        for section in report.sections
        for group in section.groups
        for path in group.images
    )


def test_build_report_skips_fake_global_for_single_sample(tmp_path: Path) -> None:
    config = AppConfig(experiment_name="single")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    explanation = ExplanationResult(
        attributions=torch.rand(1, 1, 4, 4),
        inputs=torch.rand(1, 1, 4, 4),
        run_dir=tmp_path / "transparency" / "exp",
        experiment_name="single",
        explainer_target="t",
        algorithm="IntegratedGradients",
        explainer_name="captum_ig",
        visualisers=[ConfiguredVisualiser(visualiser=_LocalImageVisualiser())],
    )
    outputs = RunOutputs(
        explanations=[explanation],
        visualisations=[],
        metrics=None,
        forward_output=torch.tensor([[0.1, 0.9]]),
        prediction_summaries=(
            PredictionSummary(sample_index=0, predicted_class=1, confidence=0.9, correct=None),
        ),
    )

    report = build_report(config, outputs)

    assert [section.title for section in report.sections] == ["Local Explanations"]


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


def test_reporting_configs_compose_sweep_report_controls() -> None:
    cfg = _compose_raitap_config()
    assert cfg.reporting.sweep_report is True
    assert cfg.hydra.callbacks.reporting_sweep._target_.endswith("ReportingSweepCallback")

    disabled_cfg = _compose_raitap_config(["reporting=disabled"])
    assert disabled_cfg.reporting._target_ is None
    assert disabled_cfg.reporting.sweep_report is False
    # The root callback remains registered; runtime guards suppress work when reporting is off.
    assert disabled_cfg.hydra.callbacks.reporting_sweep._target_.endswith("ReportingSweepCallback")

    legacy_null_cfg = OmegaConf.load(_configs_dir() / "reporting" / "null.yaml")
    assert legacy_null_cfg._target_ is None
    assert legacy_null_cfg.sweep_report is False

    opt_out_cfg = _compose_raitap_config(["reporting.sweep_report=false"])
    assert opt_out_cfg.reporting._target_ == "PDFReporter"
    assert opt_out_cfg.reporting.sweep_report is False


def test_reporting_sweep_callback_skips_when_sweep_report_disabled(
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
                "sweep_report": False,
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
            "reporting": {"_target_": None, "sweep_report": False},
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


def _write_child_manifest(child_dir: Path, *, heading: str) -> None:
    report_dir = child_dir / "reports"
    report_dir.mkdir(parents=True)
    asset = _write_test_image(report_dir / "_assets" / "child.png")
    manifest = ReportManifest(
        kind="run",
        sections=(
            ReportSection.from_groups(
                "Metrics",
                [
                    ReportGroup(
                        heading=heading,
                        images=(asset,),
                        metadata={"role": "metrics"},
                    )
                ],
            ),
        ),
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
