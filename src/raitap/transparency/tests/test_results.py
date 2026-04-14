from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import torch

from raitap.configs import resolve_run_dir, set_output_root
from raitap.configs.schema import AppConfig
from raitap.transparency.explainers.base_explainer import AttributionOnlyExplainer
from raitap.transparency.results import (
    ConfiguredVisualiser,
    ExplanationResult,
    VisualisationResult,
    _serialisable,
)
from raitap.transparency.visualisers import ShapImageVisualiser
from raitap.transparency.visualisers.base_visualiser import BaseVisualiser

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch
    from matplotlib.figure import Figure


def _make_explanation(run_dir: Path, *, explainer_name: str | None = "exp") -> ExplanationResult:
    return ExplanationResult(
        attributions=torch.randn(1, 3, 4, 4),
        inputs=torch.randn(1, 3, 4, 4),
        run_dir=run_dir,
        experiment_name="test-exp",
        explainer_target="raitap.transparency.CaptumExplainer",
        algorithm="Saliency",
        explainer_name=explainer_name,
    )


class _IdentityExplainer(AttributionOnlyExplainer):
    def compute_attributions(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del model, kwargs
        return inputs.clone()


def test_resolve_run_dir_uses_hydra_runtime(monkeypatch: MonkeyPatch) -> None:
    class _HydraRuntime:
        output_dir = "hydra-output"

    class _HydraConfig:
        runtime = _HydraRuntime()

    monkeypatch.setattr("raitap.configs.utils.HydraConfig.get", lambda: _HydraConfig())
    assert resolve_run_dir(subdir="transparency") == Path("hydra-output") / "transparency"


def test_resolve_run_dir_falls_back_to_config_output_dir(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "raitap.configs.utils.HydraConfig.get",
        lambda: (_ for _ in ()).throw(ValueError("not initialised")),
    )
    config = AppConfig()
    set_output_root(config, tmp_path)
    assert resolve_run_dir(config, subdir="transparency") == tmp_path / "transparency"


def test_base_explainer_uses_unified_output_root(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "raitap.configs.utils.HydraConfig.get",
        lambda: (_ for _ in ()).throw(ValueError("not initialised")),
    )
    explainer = _IdentityExplainer()
    inputs = torch.randn(1, 3, 4, 4)

    result = explainer.explain(
        torch.nn.Identity(),
        inputs,
        output_root=tmp_path,
    )

    assert result.run_dir == tmp_path / "transparency"
    assert (tmp_path / "transparency" / "attributions.pt").exists()


def test_serialisable_handles_runtimeerror_from_item() -> None:
    tensor = torch.randn(2, 2)
    result = _serialisable(tensor)
    assert isinstance(result, str)
    assert "tensor" in result


def test_serialisable_converts_nested_set_and_dict() -> None:
    value = {"tags": {"a", "b"}, "nested": {"k": 1}}
    result = _serialisable(value)
    assert sorted(result["tags"]) == ["a", "b"]
    assert result["nested"]["k"] == 1


def test_explanation_log_no_visualisers_logs_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "exp"
    explanation = _make_explanation(run_dir)
    explanation.write_artifacts()
    tracker = MagicMock()

    explanation.log(tracker, artifact_path="transparency", use_subdirectory=True)

    tracker.log_artifacts.assert_called_once_with(
        run_dir,
        target_subdirectory="transparency/exp",
    )


def test_explanation_log_with_visualisers_stages_explanation_only(tmp_path: Path) -> None:
    run_dir = tmp_path / "exp"
    explanation = _make_explanation(run_dir)
    explanation.write_artifacts()
    explanation.visualiser_targets = ["raitap.transparency.CaptumImageVisualiser"]
    explanation._metadata = MagicMock(return_value={"visualisers": []})  # type: ignore[method-assign]
    tracker = MagicMock()

    explanation.log(tracker, artifact_path="transparency", use_subdirectory=False)

    explanation._metadata.assert_called_once_with(visualiser_targets=[])  # type: ignore[attr-defined]
    tracker.log_artifacts.assert_called_once()


def test_explanation_log_uses_run_dir_name_fallback_when_explainer_name_is_unset(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run123"
    explanation = _make_explanation(run_dir, explainer_name=None)
    explanation.write_artifacts()
    tracker = MagicMock()

    explanation.log(tracker, artifact_path="transparency", use_subdirectory=True)

    tracker.log_artifacts.assert_called_once_with(
        run_dir,
        target_subdirectory=f"transparency/{run_dir.name}",
    )


def test_visualisation_log_uses_same_fallback_as_explanation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run123"
    explanation = _make_explanation(run_dir, explainer_name=None)

    output_path = tmp_path / "vis.png"
    output_path.write_bytes(b"not-a-real-png-but-a-file")

    visualisation = VisualisationResult(
        explanation=explanation,
        figure=MagicMock(),
        visualiser_name="SomeVisualiser",
        visualiser_target="some.target",
        output_path=output_path,
    )

    tracker = MagicMock()
    visualisation.log(tracker, artifact_path="transparency", use_subdirectory=True)

    assert tracker.log_artifacts.call_count == 1
    args, kwargs = tracker.log_artifacts.call_args
    assert isinstance(args[0], Path)
    assert args[0].name == run_dir.name
    assert kwargs["target_subdirectory"] == f"transparency/{run_dir.name}"


def test_explanation_visualise_merges_inputs_and_attributions_from_kwargs(
    tmp_path: Path,
) -> None:
    """Config/call-site may pass inputs/attributions without duplicating keyword args."""

    default_attr = torch.tensor([1.0])
    default_inp = torch.tensor([2.0])
    override_attr = torch.tensor([3.0])
    override_inp = torch.tensor([4.0])

    class _RecordingVisualiser(BaseVisualiser):
        def __init__(self) -> None:
            self.last: tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]] | None = None

        def visualise(
            self,
            attributions: torch.Tensor,
            inputs: torch.Tensor | None = None,
            **kw: Any,
        ) -> Figure:
            self.last = (attributions, inputs, dict(kw))
            fig, _ax = plt.subplots(figsize=(1, 1))
            return fig

    run_dir = tmp_path / "exp"
    explanation = ExplanationResult(
        attributions=default_attr,
        inputs=default_inp,
        run_dir=run_dir,
        experiment_name="e",
        explainer_target="t",
        algorithm="a",
        visualisers=[
            ConfiguredVisualiser(
                visualiser=_RecordingVisualiser(),
                call_kwargs={
                    "inputs": override_inp,
                    "attributions": override_attr,
                    "title": "x",
                },
            )
        ],
    )
    explanation.write_artifacts()

    explanation.visualise()

    vis = explanation.visualisers[0].visualiser
    assert isinstance(vis, _RecordingVisualiser)
    assert vis.last is not None
    attr, inp, extra = vis.last
    assert torch.equal(attr, override_attr)
    assert inp is not None and torch.equal(inp, override_inp)
    assert extra == {"title": "x"}

    explanation2 = _make_explanation(tmp_path / "exp2")
    explanation2.write_artifacts()
    rec = _RecordingVisualiser()
    explanation2.visualisers = [ConfiguredVisualiser(visualiser=rec, call_kwargs={})]
    explanation2.visualise(inputs=override_inp)
    assert rec.last is not None
    _a, inp2, _e = rec.last
    assert inp2 is not None and torch.equal(inp2, override_inp)


def test_explanation_visualise_adds_generic_sample_name_title_when_opted_in(tmp_path: Path) -> None:
    class _RecordingVisualiser(BaseVisualiser):
        def __init__(self) -> None:
            self.last_kwargs: dict[str, Any] = {}

        def visualise(
            self,
            attributions: torch.Tensor,
            inputs: torch.Tensor | None = None,
            **kw: Any,
        ) -> Figure:
            del attributions, inputs
            self.last_kwargs = dict(kw)
            fig, _ax = plt.subplots(figsize=(1, 1))
            return fig

    run_dir = tmp_path / "exp_generic_title"
    rec = _RecordingVisualiser()
    explanation = ExplanationResult(
        attributions=torch.randn(2, 3, 4, 4),
        inputs=torch.randn(2, 3, 4, 4),
        run_dir=run_dir,
        experiment_name="e",
        explainer_target="t",
        algorithm="a",
        visualisers=[
            ConfiguredVisualiser(
                visualiser=rec,
                call_kwargs={"show_sample_names": True},
            )
        ],
        kwargs={"sample_names": ["ISIC_1", "ISIC_2"]},
    )
    explanation.write_artifacts()

    [result] = explanation.visualise()

    assert rec.last_kwargs == {}
    assert result.figure.get_suptitle() == "ISIC_1 (+1)"


def test_explanation_visualise_trims_sample_names_for_shorter_batch(tmp_path: Path) -> None:
    class _SampleNameVisualiser(BaseVisualiser):
        def __init__(self) -> None:
            self.received_names: list[str] = []

        def visualise(
            self,
            attributions: torch.Tensor,
            inputs: torch.Tensor | None = None,
            sample_names: list[str] | None = None,
            show_sample_names: bool = False,
            **kw: Any,
        ) -> Figure:
            del attributions, inputs, kw
            self.received_names = [] if sample_names is None else list(sample_names)
            fig, _ax = plt.subplots(figsize=(1, 1))
            if show_sample_names and self.received_names:
                fig.suptitle(self.received_names[0])
            return fig

    run_dir = tmp_path / "exp_trim_names"
    vis = _SampleNameVisualiser()
    explanation = ExplanationResult(
        attributions=torch.randn(1, 3, 4, 4),
        inputs=torch.randn(1, 3, 4, 4),
        run_dir=run_dir,
        experiment_name="e",
        explainer_target="t",
        algorithm="a",
        visualisers=[
            ConfiguredVisualiser(
                visualiser=vis,
                call_kwargs={"show_sample_names": True},
            )
        ],
        kwargs={"sample_names": ["ISIC_1", "ISIC_2", "ISIC_3"]},
    )
    explanation.write_artifacts()

    explanation.visualise()

    assert vis.received_names == ["ISIC_1"]


def test_explanation_visualise_forwards_algorithm_when_supported(tmp_path: Path) -> None:
    class _AlgorithmVisualiser(BaseVisualiser):
        def __init__(self) -> None:
            self.last_algorithm: str | None = None
            self.last_kwargs: dict[str, Any] = {}

        def visualise(
            self,
            attributions: torch.Tensor,
            inputs: torch.Tensor | None = None,
            algorithm: str | None = None,
            **kw: Any,
        ) -> Figure:
            del attributions, inputs
            self.last_algorithm = algorithm
            self.last_kwargs = dict(kw)
            fig, _ax = plt.subplots(figsize=(1, 1))
            return fig

    run_dir = tmp_path / "exp_algorithm_forward"
    vis = _AlgorithmVisualiser()
    explanation = ExplanationResult(
        attributions=torch.randn(1, 3, 4, 4),
        inputs=torch.randn(1, 3, 4, 4),
        run_dir=run_dir,
        experiment_name="e",
        explainer_target="t",
        algorithm="IntegratedGradients",
        visualisers=[ConfiguredVisualiser(visualiser=vis)],
    )
    explanation.write_artifacts()

    explanation.visualise()

    assert vis.last_algorithm == "IntegratedGradients"
    assert vis.last_kwargs == {}


def test_explanation_visualise_sets_shap_image_default_title_from_algorithm(tmp_path: Path) -> None:
    run_dir = tmp_path / "exp_shap_title"
    explanation = ExplanationResult(
        attributions=torch.randn(1, 3, 8, 8),
        inputs=torch.rand(1, 3, 8, 8),
        run_dir=run_dir,
        experiment_name="e",
        explainer_target="t",
        algorithm="GradientExplainer",
        visualisers=[ConfiguredVisualiser(visualiser=ShapImageVisualiser(show_colorbar=False))],
    )
    explanation.write_artifacts()

    [result] = explanation.visualise()

    titles = [ax.get_title() for ax in result.figure.axes if ax.get_title()]
    assert titles == ["Original Image", "GradientExplainer (SHAP)"]


def test_explanation_visualise_preserves_shap_constructor_title(tmp_path: Path) -> None:
    run_dir = tmp_path / "exp_shap_constructor_title"
    explanation = ExplanationResult(
        attributions=torch.randn(1, 3, 8, 8),
        inputs=torch.rand(1, 3, 8, 8),
        run_dir=run_dir,
        experiment_name="e",
        explainer_target="t",
        algorithm="GradientExplainer",
        visualisers=[
            ConfiguredVisualiser(
                visualiser=ShapImageVisualiser(title="Configured SHAP", show_colorbar=False)
            )
        ],
    )
    explanation.write_artifacts()

    [result] = explanation.visualise()

    titles = [ax.get_title() for ax in result.figure.axes if ax.get_title()]
    assert titles == ["Original Image", "Configured SHAP"]


def test_explanation_visualise_preserves_explicit_shap_image_title(tmp_path: Path) -> None:
    run_dir = tmp_path / "exp_shap_explicit_title"
    explanation = ExplanationResult(
        attributions=torch.randn(1, 3, 8, 8),
        inputs=torch.rand(1, 3, 8, 8),
        run_dir=run_dir,
        experiment_name="e",
        explainer_target="t",
        algorithm="GradientExplainer",
        visualisers=[
            ConfiguredVisualiser(
                visualiser=ShapImageVisualiser(title="Configured SHAP", show_colorbar=False),
            )
        ],
    )
    explanation.write_artifacts()

    [result] = explanation.visualise(title="Runtime SHAP")

    titles = [ax.get_title() for ax in result.figure.axes if ax.get_title()]
    assert titles == ["Original Image", "Runtime SHAP"]


def test_explanation_visualise_forwards_algorithm_to_shap_subclasses(tmp_path: Path) -> None:
    class _SubclassedShapImageVisualiser(ShapImageVisualiser):
        pass

    run_dir = tmp_path / "exp_shap_subclass_title"
    explanation = ExplanationResult(
        attributions=torch.randn(1, 3, 8, 8),
        inputs=torch.rand(1, 3, 8, 8),
        run_dir=run_dir,
        experiment_name="e",
        explainer_target="t",
        algorithm="DeepExplainer",
        visualisers=[
            ConfiguredVisualiser(visualiser=_SubclassedShapImageVisualiser(show_colorbar=False))
        ],
    )
    explanation.write_artifacts()

    [result] = explanation.visualise()

    titles = [ax.get_title() for ax in result.figure.axes if ax.get_title()]
    assert titles == ["Original Image", "DeepExplainer (SHAP)"]
