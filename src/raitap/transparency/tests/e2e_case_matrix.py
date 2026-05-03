from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from raitap.models.backend import TorchBackend
from raitap.transparency import ExplanationResult, VisualisationResult
from raitap.transparency.contracts import InputSpec
from raitap.transparency.explainers import CaptumExplainer, ShapExplainer
from raitap.transparency.factory import Explanation
from raitap.transparency.results import ConfiguredVisualiser
from raitap.transparency.visualisers import (
    BaseVisualiser,
    CaptumImageVisualiser,
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

    import pytest
    from matplotlib.figure import Figure

    from raitap.configs.schema import AppConfig
    from raitap.models import Model
    from raitap.models.backend import OnnxBackend

Framework = Literal["captum", "shap"]
ExecutionMode = Literal["compute", "explain", "factory"]
ModelFixtureName = Literal["simple_cnn", "simple_mlp", "simple_timeseries_model"]
InputFixtureName = Literal["sample_images", "sample_tabular", "sample_timeseries"]
NeedsFixtureName = Literal["needs_captum", "needs_shap", "needs_onnx"]
BackendFixtureName = Literal["onnx_linear_backend"]
TargetMode = int | Literal["alternating_binary", "none", "zeros_tensor"]
ShapeExpectation = Literal["same_as_inputs", "same_as_inputs_minus_last", "same_batch_size"]
RunDirMode = Literal["factory_subdir", "plain_transparency"]


@dataclass(frozen=True)
class MatrixCase:
    id: str
    framework: Framework
    mode: ExecutionMode
    algorithm: str
    needs_fixture: NeedsFixtureName
    model_fixture: ModelFixtureName
    input_fixture: InputFixtureName
    target_mode: TargetMode
    expected_shape: ShapeExpectation
    run_dir_mode: RunDirMode
    explainer_init_kwargs: dict[str, object] = field(default_factory=dict)
    explain_call_kwargs: dict[str, object] = field(default_factory=dict)
    visualiser_cls: type[BaseVisualiser] | None = None
    visualiser_kwargs: dict[str, object] = field(default_factory=dict)
    background_samples: int | None = None
    expected_metadata_kwargs: dict[str, object] = field(default_factory=dict)
    experiment_name: str | None = None
    factory_explainer_name: str | None = None
    backend_fixture: BackendFixtureName | None = None
    requires_onnx: bool = False
    mpl_baseline_filename: str | None = None
    mpl_use_deterministic_inputs: bool = False
    mpl_target_mode: TargetMode | None = None


@dataclass(frozen=True)
class BehaviorRunResult:
    case: MatrixCase
    inputs: torch.Tensor
    attributions: torch.Tensor | None
    explanation: ExplanationResult | None
    visualisations: list[VisualisationResult]
    metadata_before_visualise: dict[str, object] | None
    metadata_after_visualise: dict[str, object] | None
    saved_attributions: torch.Tensor | None


def read_metadata(run_dir: Path) -> dict[str, object]:
    return cast(
        "dict[str, object]",
        json.loads((run_dir / "metadata.json").read_text(encoding="utf-8")),
    )


def load_saved_attributions(run_dir: Path) -> torch.Tensor:
    return cast("torch.Tensor", torch.load(run_dir / "attributions.pt"))


def _framework_target_suffix(case: MatrixCase) -> str:
    if case.framework == "captum":
        return "CaptumExplainer"
    return "ShapExplainer"


def _explainer_target(case: MatrixCase) -> str:
    class_name = _framework_target_suffix(case)
    return f"raitap.transparency.{class_name}"


def _resolve_target_mode(
    target_mode: TargetMode,
    inputs: torch.Tensor,
) -> int | list[int] | torch.Tensor | None:
    if isinstance(target_mode, int):
        return target_mode
    if target_mode == "alternating_binary":
        return [index % 2 for index in range(inputs.shape[0])]
    if target_mode == "zeros_tensor":
        return torch.zeros(inputs.shape[0], dtype=torch.long)
    return None


def _resolve_target(
    case: MatrixCase, inputs: torch.Tensor
) -> int | list[int] | torch.Tensor | None:
    return _resolve_target_mode(case.target_mode, inputs)


def _expected_serialised_target(case: MatrixCase, inputs: torch.Tensor) -> int | list[int] | None:
    target = _resolve_target(case, inputs)
    if target is None:
        return None
    if isinstance(target, torch.Tensor):
        return cast("list[int]", target.tolist())
    return target


def _select_background(case: MatrixCase, inputs: torch.Tensor) -> torch.Tensor | None:
    if case.background_samples is None:
        return None
    return inputs[: case.background_samples]


def _build_visualisers(case: MatrixCase) -> list[ConfiguredVisualiser]:
    if case.visualiser_cls is None:
        return []
    return [
        ConfiguredVisualiser(
            case.visualiser_cls(**case.visualiser_kwargs),
        )
    ]


def _build_factory_config(case: MatrixCase, tmp_path: Path) -> AppConfig:
    explainer_name = case.factory_explainer_name
    if explainer_name is None:
        raise ValueError(f"{case.id} requires a factory explainer name.")

    visualisers: list[dict[str, object]] = []
    if case.visualiser_cls is not None:
        visualiser_entry: dict[str, object] = {
            "_target_": f"raitap.transparency.{case.visualiser_cls.__name__}"
        }
        if case.visualiser_kwargs:
            visualiser_entry["constructor"] = dict(case.visualiser_kwargs)
        visualisers.append(visualiser_entry)

    call_kwargs: dict[str, object] = dict(case.explain_call_kwargs)
    target = case.target_mode
    if isinstance(target, int):
        call_kwargs["target"] = target

    return cast(
        "AppConfig",
        cast(
            "object",
            SimpleNamespace(
                experiment_name=case.experiment_name or "test_e2e",
                _output_root=str(tmp_path),
                transparency={
                    explainer_name: OmegaConf.create(
                        {
                            "_target_": _explainer_target(case),
                            "algorithm": case.algorithm,
                            "constructor": dict(case.explainer_init_kwargs),
                            "call": call_kwargs,
                            "visualisers": visualisers,
                        }
                    )
                },
            ),
        ),
    )


def _make_explainer(case: MatrixCase) -> CaptumExplainer | ShapExplainer:
    if case.framework == "captum":
        return CaptumExplainer(case.algorithm, **case.explainer_init_kwargs)
    return ShapExplainer(case.algorithm, **case.explainer_init_kwargs)


def _fetch_model(request: pytest.FixtureRequest, case: MatrixCase) -> nn.Module:
    return cast("nn.Module", request.getfixturevalue(case.model_fixture))


def _fetch_inputs(request: pytest.FixtureRequest, case: MatrixCase) -> torch.Tensor:
    return cast("torch.Tensor", request.getfixturevalue(case.input_fixture))


def _fetch_backend(
    request: pytest.FixtureRequest,
    case: MatrixCase,
) -> OnnxBackend | None:
    if case.backend_fixture is None:
        return None
    return cast("OnnxBackend", request.getfixturevalue(case.backend_fixture))


def _assert_tensor_shape(
    tensor: torch.Tensor,
    inputs: torch.Tensor,
    *,
    expectation: ShapeExpectation,
) -> None:
    if expectation == "same_as_inputs":
        assert tensor.shape == inputs.shape
        return
    if expectation == "same_batch_size":
        assert tensor.shape[0] == inputs.shape[0]
        return
    assert tensor.shape[:-1] == inputs.shape


def _compute_case_attributions(
    case: MatrixCase,
    *,
    model: nn.Module,
    inputs: torch.Tensor,
    target: int | list[int] | torch.Tensor | None,
    background: torch.Tensor | None,
) -> torch.Tensor:
    explainer = _make_explainer(case)
    if case.explain_call_kwargs:
        raise ValueError(f"{case.id} defines explain_call_kwargs but runs in compute mode.")
    if case.framework == "captum":
        assert isinstance(explainer, CaptumExplainer)
        return explainer.compute_attributions(model, inputs, target=target)
    assert isinstance(explainer, ShapExplainer)
    return explainer.compute_attributions(
        model,
        inputs,
        background_data=background,
        target=target,
    )


def _run_explain_case(
    case: MatrixCase,
    *,
    model: nn.Module,
    inputs: torch.Tensor,
    backend: OnnxBackend | None,
    target: int | list[int] | torch.Tensor | None,
    background: torch.Tensor | None,
    tmp_path: Path,
) -> ExplanationResult:
    explainer = _make_explainer(case)
    visualisers = _build_visualisers(case)
    run_dir = expected_run_dir(case, tmp_path)
    extra_call_kwargs: dict[str, Any] = dict(case.explain_call_kwargs)
    runtime_model: nn.Module = model if backend is None else backend.as_model_for_explanation()

    if backend is not None:
        explainer.check_backend_compat(backend)
        extra_call_kwargs["backend"] = backend

    raitap_kwargs = dict(cast("dict[str, object]", extra_call_kwargs.pop("raitap_kwargs", {})))
    if "batch_size" in extra_call_kwargs:
        raitap_kwargs.setdefault("batch_size", extra_call_kwargs.pop("batch_size"))
    raitap_kwargs.setdefault("input_metadata", _input_metadata_for_case(case, inputs))
    extra_call_kwargs["raitap_kwargs"] = raitap_kwargs

    if case.framework == "captum":
        return explainer.explain(
            runtime_model,
            inputs,
            run_dir=run_dir,
            target=target,
            visualisers=visualisers,
            **extra_call_kwargs,
        )

    return explainer.explain(
        runtime_model,
        inputs,
        run_dir=run_dir,
        background_data=background,
        target=target,
        visualisers=visualisers,
        **extra_call_kwargs,
    )


def _assert_metadata_invariants(
    metadata: dict[str, object],
    *,
    case: MatrixCase,
    has_visualisers: bool,
) -> None:
    assert set(metadata) == {
        "experiment_name",
        "target",
        "algorithm",
        "visualisers",
        "kwargs",
        "call_kwargs",
        "payload_kind",
        "semantics",
    }
    assert metadata["payload_kind"] == "attributions"
    semantics = cast("dict[str, object]", metadata["semantics"])
    assert semantics["scope"] == "local"
    assert semantics["scope_definition_step"] == "explainer_output"
    assert semantics["payload_kind"] == "attributions"
    assert "output_space" in semantics
    assert metadata["experiment_name"] == case.experiment_name
    assert metadata["algorithm"] == case.algorithm
    assert str(metadata["target"]).endswith(_framework_target_suffix(case))
    assert isinstance(metadata["kwargs"], dict)
    assert isinstance(metadata["call_kwargs"], dict)
    assert isinstance(metadata["visualisers"], list)
    if has_visualisers:
        assert len(cast("list[str]", metadata["visualisers"])) >= 1
    else:
        assert metadata["visualisers"] == []


def expected_run_dir(case: MatrixCase, tmp_path: Path) -> Path:
    if case.run_dir_mode == "factory_subdir":
        explainer_name = case.factory_explainer_name
        if explainer_name is None:
            raise ValueError(f"{case.id} requires a factory explainer name.")
        return tmp_path / "transparency" / explainer_name
    return tmp_path / "transparency"


def _input_metadata_for_case(case: MatrixCase, inputs: torch.Tensor) -> InputSpec:
    shape = tuple(int(dim) for dim in inputs.shape)
    if case.input_fixture == "sample_images":
        return InputSpec(
            kind="image",
            shape=shape,
            layout="NCHW",
            metadata={"kind": "image", "layout": "NCHW"},
        )
    if case.input_fixture == "sample_tabular":
        return InputSpec(
            kind="tabular",
            shape=shape,
            layout="(B,F)",
            metadata={"kind": "tabular", "layout": "(B,F)"},
        )
    return InputSpec(
        kind="time_series",
        shape=shape,
        layout="(B,T,C)",
        metadata={"kind": "time_series", "layout": "(B,T,C)"},
    )


def run_behavior_case(
    case: MatrixCase,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> BehaviorRunResult:
    request.getfixturevalue(case.needs_fixture)
    if case.requires_onnx:
        request.getfixturevalue("needs_onnx")
    model = _fetch_model(request, case)
    inputs = _fetch_inputs(request, case)
    backend = _fetch_backend(request, case)
    target = _resolve_target(case, inputs)
    background = _select_background(case, inputs)

    if case.mode == "compute":
        if backend is not None:
            raise ValueError(
                f"{case.id} uses an ONNX backend but compute mode is not supported here."
            )
        attributions = _compute_case_attributions(
            case,
            model=model,
            inputs=inputs,
            target=target,
            background=background,
        )
        return BehaviorRunResult(
            case=case,
            inputs=inputs,
            attributions=attributions,
            explanation=None,
            visualisations=[],
            metadata_before_visualise=None,
            metadata_after_visualise=None,
            saved_attributions=None,
        )

    if case.mode == "factory":
        config = _build_factory_config(case, tmp_path)
        model_wrapper = cast("Model", SimpleNamespace(backend=TorchBackend(model)))
        input_metadata = _input_metadata_for_case(case, inputs)
        if background is None:
            explanation = Explanation(
                config,
                cast("str", case.factory_explainer_name),
                model_wrapper,
                inputs,
                input_metadata=input_metadata,
            )
        else:
            explanation = Explanation(
                config,
                cast("str", case.factory_explainer_name),
                model_wrapper,
                inputs,
                input_metadata=input_metadata,
                background_data=background,
            )
    else:
        explanation = _run_explain_case(
            case,
            model=model,
            inputs=inputs,
            backend=backend,
            target=target,
            background=background,
            tmp_path=tmp_path,
        )

    metadata_before_visualise = read_metadata(explanation.run_dir)
    saved_attributions = load_saved_attributions(explanation.run_dir)
    visualisations = explanation.visualise()
    metadata_after_visualise = read_metadata(explanation.run_dir)

    return BehaviorRunResult(
        case=case,
        inputs=inputs,
        attributions=explanation.attributions,
        explanation=explanation,
        visualisations=visualisations,
        metadata_before_visualise=metadata_before_visualise,
        metadata_after_visualise=metadata_after_visualise,
        saved_attributions=saved_attributions,
    )


def assert_behavior_case(
    result: BehaviorRunResult,
    *,
    tmp_path: Path,
) -> None:
    case = result.case

    if case.mode == "compute":
        attributions = result.attributions
        assert attributions is not None
        assert isinstance(attributions, torch.Tensor)
        _assert_tensor_shape(attributions, result.inputs, expectation=case.expected_shape)
        return

    explanation = result.explanation
    metadata_before = result.metadata_before_visualise
    metadata_after = result.metadata_after_visualise
    saved_attributions = result.saved_attributions

    assert explanation is not None
    assert metadata_before is not None
    assert metadata_after is not None
    assert saved_attributions is not None
    assert isinstance(explanation, ExplanationResult)
    _assert_tensor_shape(explanation.attributions, result.inputs, expectation=case.expected_shape)
    assert explanation.run_dir == expected_run_dir(case, tmp_path)
    assert (explanation.run_dir / "attributions.pt").exists()
    assert (explanation.run_dir / "metadata.json").exists()
    assert torch.equal(saved_attributions, explanation.attributions)

    _assert_metadata_invariants(metadata_before, case=case, has_visualisers=False)
    _assert_metadata_invariants(
        metadata_after, case=case, has_visualisers=case.visualiser_cls is not None
    )

    expected_target = _expected_serialised_target(case, result.inputs)
    metadata_call_kwargs = cast("dict[str, object]", metadata_after["call_kwargs"])
    if expected_target is not None:
        assert metadata_call_kwargs["target"] == expected_target

    for key, value in case.expected_metadata_kwargs.items():
        assert metadata_call_kwargs[key] == value

    if case.visualiser_cls is None:
        assert result.visualisations == []
        return

    assert len(result.visualisations) == 1
    assert isinstance(result.visualisations[0], VisualisationResult)
    assert cast("list[str]", metadata_after["visualisers"])[0].endswith(
        f"{case.visualiser_cls.__name__}_0"
    )
    assert (
        result.visualisations[0].output_path
        == explanation.run_dir / f"{case.visualiser_cls.__name__}_0.png"
    )
    assert result.visualisations[0].output_path.exists()


def build_deterministic_cnn() -> nn.Module:
    model = nn.Sequential(
        nn.Conv2d(3, 2, kernel_size=1, bias=True),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(2, 2, bias=True),
    )
    with torch.no_grad():
        conv = cast("nn.Conv2d", model[0])
        linear = cast("nn.Linear", model[4])
        conv.weight.copy_(
            torch.tensor(
                [
                    [[[0.70]], [[-0.25]], [[0.15]]],
                    [[[-0.35]], [[0.55]], [[0.40]]],
                ],
                dtype=torch.float32,
            )
        )
        if conv.bias is None or linear.bias is None:
            raise ValueError("The deterministic MPL baseline model requires bias parameters.")
        conv.bias.copy_(torch.tensor([0.05, -0.10], dtype=torch.float32))
        linear.weight.copy_(torch.tensor([[0.60, -0.45], [-0.30, 0.80]], dtype=torch.float32))
        linear.bias.copy_(torch.tensor([0.10, -0.05], dtype=torch.float32))

    model.eval()
    return model


def literal_image_batch() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [
                    [0.00, 0.10, 0.20, 0.30],
                    [0.05, 0.15, 0.25, 0.35],
                    [0.10, 0.20, 0.30, 0.40],
                    [0.15, 0.25, 0.35, 0.45],
                ],
                [
                    [0.45, 0.35, 0.25, 0.15],
                    [0.40, 0.30, 0.20, 0.10],
                    [0.35, 0.25, 0.15, 0.05],
                    [0.30, 0.20, 0.10, 0.00],
                ],
                [
                    [0.20, 0.30, 0.40, 0.50],
                    [0.25, 0.35, 0.45, 0.55],
                    [0.30, 0.40, 0.50, 0.60],
                    [0.35, 0.45, 0.55, 0.65],
                ],
            ]
        ],
        dtype=torch.float32,
    )


def render_mpl_figure(
    case: MatrixCase,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> Figure:
    request.getfixturevalue(case.needs_fixture)
    if case.requires_onnx:
        request.getfixturevalue("needs_onnx")
    model = (
        build_deterministic_cnn()
        if case.mpl_use_deterministic_inputs
        else _fetch_model(request, case)
    )
    inputs = (
        literal_image_batch() if case.mpl_use_deterministic_inputs else _fetch_inputs(request, case)
    )
    mpl_target_mode = case.mpl_target_mode if case.mpl_target_mode is not None else case.target_mode
    target = _resolve_target_mode(mpl_target_mode, inputs)
    background = _select_background(case, inputs)

    if case.mode == "explain" and case.explain_call_kwargs:
        # Route through _run_explain_case so explain_call_kwargs (e.g. Occlusion's
        # sliding_window_shapes / strides) are forwarded automatically.
        backend = _fetch_backend(request, case)
        explanation = _run_explain_case(
            case,
            model=model,
            inputs=inputs,
            backend=backend,
            target=target,
            background=background,
            tmp_path=tmp_path,
        )
        attributions = explanation.attributions
    else:
        # mode="compute", mode="factory", and mode="explain" without extra call kwargs:
        # use the direct compute path so existing PNG baselines remain valid and
        # non-deterministic backends (e.g. SHAP) stay on the same code path.
        attributions = _compute_case_attributions(
            case,
            model=model,
            inputs=inputs,
            target=target,
            background=background,
        )

    if case.visualiser_cls is None:
        raise ValueError(f"{case.id} does not define a visualiser for MPL rendering.")

    visualiser = case.visualiser_cls(**case.visualiser_kwargs)
    if isinstance(visualiser, CaptumImageVisualiser):
        figure = visualiser.visualise(attributions, inputs=inputs, max_samples=1)
    else:
        figure = visualiser.visualise(attributions, inputs=inputs)
    figure.set_size_inches(4.0, 4.0)
    figure.tight_layout()
    return figure


MATRIX_CASES: tuple[MatrixCase, ...] = (
    MatrixCase(
        id="captum_smoke_config_image",
        framework="captum",
        mode="factory",
        algorithm="IntegratedGradients",
        needs_fixture="needs_captum",
        model_fixture="simple_cnn",
        input_fixture="sample_images",
        target_mode=0,
        expected_shape="same_as_inputs",
        run_dir_mode="factory_subdir",
        visualiser_cls=CaptumImageVisualiser,
        visualiser_kwargs={
            "method": "heat_map",
            "show_colorbar": False,
            "include_original_image": False,
        },
        experiment_name="test_captum_e2e",
        factory_explainer_name="captum_smoke",
        mpl_baseline_filename="captum_ig_image_heat_map.png",
        mpl_use_deterministic_inputs=True,
        mpl_target_mode=1,
    ),
    MatrixCase(
        id="shap_gradient_image_mpl_baseline",
        framework="shap",
        mode="explain",
        algorithm="GradientExplainer",
        needs_fixture="needs_shap",
        model_fixture="simple_cnn",
        input_fixture="sample_images",
        target_mode=0,
        expected_shape="same_as_inputs",
        run_dir_mode="plain_transparency",
        visualiser_cls=ShapImageVisualiser,
        background_samples=2,
        mpl_baseline_filename="shap_gradient_image_heat_map.png",
        mpl_use_deterministic_inputs=True,
        mpl_target_mode=1,
    ),
    MatrixCase(
        id="captum_saliency_image_heat_map",
        framework="captum",
        mode="explain",
        algorithm="Saliency",
        needs_fixture="needs_captum",
        model_fixture="simple_cnn",
        input_fixture="sample_images",
        target_mode=0,
        expected_shape="same_as_inputs",
        run_dir_mode="plain_transparency",
        visualiser_cls=CaptumImageVisualiser,
        visualiser_kwargs={
            "method": "heat_map",
            "show_colorbar": False,
            "include_original_image": False,
        },
    ),
    MatrixCase(
        id="captum_occlusion_image_heat_map",
        framework="captum",
        mode="explain",
        algorithm="Occlusion",
        needs_fixture="needs_captum",
        model_fixture="simple_cnn",
        input_fixture="sample_images",
        target_mode=0,
        expected_shape="same_as_inputs",
        run_dir_mode="plain_transparency",
        explain_call_kwargs={
            # Use a sub-image occlusion window so the deterministic MPL baseline
            # produces spatial structure instead of a uniform full-frame heat map.
            "sliding_window_shapes": [3, 2, 2],
            "strides": [1, 1, 1],
            "perturbations_per_eval": 2,
        },
        expected_metadata_kwargs={
            "sliding_window_shapes": [3, 2, 2],
            "strides": [1, 1, 1],
            "perturbations_per_eval": 2,
        },
        visualiser_cls=CaptumImageVisualiser,
        visualiser_kwargs={
            "method": "heat_map",
            "show_colorbar": False,
            "include_original_image": False,
        },
        mpl_baseline_filename="captum_occlusion_image_heat_map.png",
        mpl_use_deterministic_inputs=True,
        mpl_target_mode=1,
    ),
    MatrixCase(
        id="captum_saliency_image_batch_targets",
        framework="captum",
        mode="compute",
        algorithm="Saliency",
        needs_fixture="needs_captum",
        model_fixture="simple_cnn",
        input_fixture="sample_images",
        target_mode="alternating_binary",
        expected_shape="same_as_inputs",
        run_dir_mode="plain_transparency",
    ),
    MatrixCase(
        id="captum_saliency_tabular_bar_chart",
        framework="captum",
        mode="explain",
        algorithm="Saliency",
        needs_fixture="needs_captum",
        model_fixture="simple_mlp",
        input_fixture="sample_tabular",
        target_mode=0,
        expected_shape="same_as_inputs",
        run_dir_mode="plain_transparency",
        visualiser_cls=TabularBarChartVisualiser,
        visualiser_kwargs={"feature_names": [f"feature_{index}" for index in range(10)]},
    ),
    MatrixCase(
        id="captum_integrated_gradients_tabular_compute",
        framework="captum",
        mode="compute",
        algorithm="IntegratedGradients",
        needs_fixture="needs_captum",
        model_fixture="simple_mlp",
        input_fixture="sample_tabular",
        target_mode=0,
        expected_shape="same_as_inputs",
        run_dir_mode="plain_transparency",
    ),
    MatrixCase(
        id="captum_factory_saliency_image",
        framework="captum",
        mode="factory",
        algorithm="Saliency",
        needs_fixture="needs_captum",
        model_fixture="simple_cnn",
        input_fixture="sample_images",
        target_mode=0,
        expected_shape="same_as_inputs",
        run_dir_mode="factory_subdir",
        visualiser_cls=CaptumImageVisualiser,
        visualiser_kwargs={
            "method": "blended_heat_map",
            "show_colorbar": False,
            "include_original_image": False,
        },
        experiment_name="test_captum_factory_e2e",
        factory_explainer_name="captum_saliency",
    ),
    MatrixCase(
        id="captum_layer_gradcam_masked_image",
        framework="captum",
        mode="explain",
        algorithm="LayerGradCam",
        needs_fixture="needs_captum",
        model_fixture="simple_cnn",
        input_fixture="sample_images",
        target_mode=0,
        expected_shape="same_batch_size",
        run_dir_mode="plain_transparency",
        explainer_init_kwargs={"layer_path": "0"},
        visualiser_cls=CaptumImageVisualiser,
        visualiser_kwargs={
            "method": "masked_image",
            "sign": "absolute_value",
            "show_colorbar": False,
            "include_original_image": False,
        },
    ),
    MatrixCase(
        id="captum_saliency_timeseries_overlay",
        framework="captum",
        mode="explain",
        algorithm="Saliency",
        needs_fixture="needs_captum",
        model_fixture="simple_timeseries_model",
        input_fixture="sample_timeseries",
        target_mode=0,
        expected_shape="same_as_inputs",
        run_dir_mode="plain_transparency",
        visualiser_cls=CaptumTimeSeriesVisualiser,
        visualiser_kwargs={"method": "overlay_individual", "sign": "absolute_value"},
    ),
    MatrixCase(
        id="captum_integrated_gradients_timeseries_overlay_combined",
        framework="captum",
        mode="explain",
        algorithm="IntegratedGradients",
        needs_fixture="needs_captum",
        model_fixture="simple_timeseries_model",
        input_fixture="sample_timeseries",
        target_mode=0,
        expected_shape="same_as_inputs",
        run_dir_mode="plain_transparency",
        visualiser_cls=CaptumTimeSeriesVisualiser,
        visualiser_kwargs={"method": "overlay_combined", "sign": "absolute_value"},
    ),
    MatrixCase(
        id="captum_feature_ablation_onnx_explain",
        framework="captum",
        mode="explain",
        algorithm="FeatureAblation",
        needs_fixture="needs_captum",
        model_fixture="simple_mlp",
        input_fixture="sample_tabular",
        target_mode=0,
        expected_shape="same_as_inputs",
        run_dir_mode="plain_transparency",
        backend_fixture="onnx_linear_backend",
        requires_onnx=True,
    ),
    MatrixCase(
        id="captum_feature_ablation_onnx_batched_explain",
        framework="captum",
        mode="explain",
        algorithm="FeatureAblation",
        needs_fixture="needs_captum",
        model_fixture="simple_mlp",
        input_fixture="sample_tabular",
        target_mode=0,
        expected_shape="same_as_inputs",
        run_dir_mode="plain_transparency",
        backend_fixture="onnx_linear_backend",
        requires_onnx=True,
        explain_call_kwargs={"batch_size": 2},
    ),
    MatrixCase(
        id="shap_deep_image_pipeline",
        framework="shap",
        mode="explain",
        algorithm="DeepExplainer",
        needs_fixture="needs_shap",
        model_fixture="simple_cnn",
        input_fixture="sample_images",
        target_mode=0,
        expected_shape="same_as_inputs",
        run_dir_mode="plain_transparency",
        visualiser_cls=ShapImageVisualiser,
        background_samples=2,
    ),
    MatrixCase(
        id="shap_deep_multiclass_all_targets",
        framework="shap",
        mode="compute",
        algorithm="DeepExplainer",
        needs_fixture="needs_shap",
        model_fixture="simple_cnn",
        input_fixture="sample_images",
        target_mode="none",
        expected_shape="same_as_inputs_minus_last",
        run_dir_mode="plain_transparency",
        background_samples=2,
    ),
    MatrixCase(
        id="shap_gradient_no_background_fallback",
        framework="shap",
        mode="compute",
        algorithm="GradientExplainer",
        needs_fixture="needs_shap",
        model_fixture="simple_cnn",
        input_fixture="sample_images",
        target_mode=0,
        expected_shape="same_as_inputs",
        run_dir_mode="plain_transparency",
    ),
    MatrixCase(
        id="shap_gradient_beeswarm_list_targets",
        framework="shap",
        mode="explain",
        algorithm="GradientExplainer",
        needs_fixture="needs_shap",
        model_fixture="simple_mlp",
        input_fixture="sample_tabular",
        target_mode="alternating_binary",
        expected_shape="same_as_inputs",
        run_dir_mode="plain_transparency",
        visualiser_cls=ShapBeeswarmVisualiser,
        visualiser_kwargs={"feature_names": [f"f{index}" for index in range(10)]},
        background_samples=4,
    ),
    MatrixCase(
        id="shap_gradient_tensor_target_indexing",
        framework="shap",
        mode="compute",
        algorithm="GradientExplainer",
        needs_fixture="needs_shap",
        model_fixture="simple_mlp",
        input_fixture="sample_tabular",
        target_mode="zeros_tensor",
        expected_shape="same_as_inputs",
        run_dir_mode="plain_transparency",
        background_samples=4,
    ),
    MatrixCase(
        id="shap_gradient_waterfall_visualiser",
        framework="shap",
        mode="explain",
        algorithm="GradientExplainer",
        needs_fixture="needs_shap",
        model_fixture="simple_mlp",
        input_fixture="sample_tabular",
        target_mode=0,
        expected_shape="same_as_inputs",
        run_dir_mode="plain_transparency",
        visualiser_cls=ShapWaterfallVisualiser,
        visualiser_kwargs={
            "feature_names": [f"f{index}" for index in range(10)],
            "expected_value": 0.5,
            "sample_index": 1,
            "max_display": 5,
        },
        background_samples=4,
    ),
    MatrixCase(
        id="shap_gradient_force_visualiser",
        framework="shap",
        mode="explain",
        algorithm="GradientExplainer",
        needs_fixture="needs_shap",
        model_fixture="simple_mlp",
        input_fixture="sample_tabular",
        target_mode=0,
        expected_shape="same_as_inputs",
        run_dir_mode="plain_transparency",
        visualiser_cls=ShapForceVisualiser,
        visualiser_kwargs={
            "feature_names": [f"f{index}" for index in range(10)],
            "expected_value": 0.5,
            "sample_index": 0,
        },
        background_samples=4,
    ),
    MatrixCase(
        id="shap_gradient_bar_visualiser",
        framework="shap",
        mode="explain",
        algorithm="GradientExplainer",
        needs_fixture="needs_shap",
        model_fixture="simple_mlp",
        input_fixture="sample_tabular",
        target_mode=0,
        expected_shape="same_as_inputs",
        run_dir_mode="plain_transparency",
        visualiser_cls=ShapBarVisualiser,
        visualiser_kwargs={"feature_names": [f"f{index}" for index in range(10)]},
        background_samples=4,
    ),
    MatrixCase(
        id="shap_gradient_image_alternating_targets",
        framework="shap",
        mode="explain",
        algorithm="GradientExplainer",
        needs_fixture="needs_shap",
        model_fixture="simple_cnn",
        input_fixture="sample_images",
        target_mode="alternating_binary",
        expected_shape="same_as_inputs",
        run_dir_mode="plain_transparency",
        visualiser_cls=ShapImageVisualiser,
        background_samples=2,
    ),
    MatrixCase(
        id="shap_factory_gradient_pipeline",
        framework="shap",
        mode="factory",
        algorithm="GradientExplainer",
        needs_fixture="needs_shap",
        model_fixture="simple_cnn",
        input_fixture="sample_images",
        target_mode=0,
        expected_shape="same_as_inputs",
        run_dir_mode="factory_subdir",
        visualiser_cls=ShapImageVisualiser,
        visualiser_kwargs={},
        background_samples=2,
        experiment_name="test_shap_e2e",
        factory_explainer_name="shap_gradient",
    ),
    MatrixCase(
        id="shap_object_api_image",
        framework="shap",
        mode="explain",
        algorithm="GradientExplainer",
        needs_fixture="needs_shap",
        model_fixture="simple_cnn",
        input_fixture="sample_images",
        target_mode=0,
        expected_shape="same_as_inputs",
        run_dir_mode="plain_transparency",
        visualiser_cls=ShapImageVisualiser,
        background_samples=2,
    ),
    MatrixCase(
        id="shap_kernel_onnx_explain",
        framework="shap",
        mode="explain",
        algorithm="KernelExplainer",
        needs_fixture="needs_shap",
        model_fixture="simple_mlp",
        input_fixture="sample_tabular",
        target_mode=0,
        expected_shape="same_as_inputs",
        run_dir_mode="plain_transparency",
        backend_fixture="onnx_linear_backend",
        requires_onnx=True,
        background_samples=2,
        explain_call_kwargs={"nsamples": 10},
    ),
)
