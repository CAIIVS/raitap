from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, cast
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import pytest
import torch
from omegaconf import OmegaConf

from raitap.models.backend import ModelBackend
from raitap.transparency import (
    ExplainerBackendIncompatibilityError,
    PayloadVisualiserIncompatibilityError,
    VisualiserIncompatibilityError,
)
from raitap.transparency.contracts import ExplanationPayloadKind
from raitap.transparency.factory import (
    Explanation,
    _resolve_call_data_sources,
    check_explainer_visualiser_compat,
    create_explainer,
    create_visualisers,
)
from raitap.transparency.results import ConfiguredVisualiser, ExplanationResult
from raitap.transparency.visualisers.base_visualiser import BaseVisualiser

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from raitap.configs.schema import AppConfig


class _BackendStub(ModelBackend):
    def __init__(self, model: torch.nn.Module, *, supports_torch_autograd: bool = True) -> None:
        self._model = model
        self.supports_torch_autograd = supports_torch_autograd

    @property
    def hardware_label(self) -> str:
        return "stub"

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def _prepare_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return kwargs

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._model(inputs)

    def as_model_for_explanation(self) -> torch.nn.Module:
        return self._model


def _load_transparency_preset(name: str) -> Any:
    configs_dir = Path(__file__).resolve().parents[2] / "configs" / "transparency"
    preset = cast(
        "dict[str, Any]",
        OmegaConf.to_container(OmegaConf.load(configs_dir / f"{name}.yaml")),
    )
    return preset[name]


def _make_config(tmp_path: Path, transparency_config: Any) -> AppConfig:
    return cast(
        "AppConfig",
        SimpleNamespace(
            experiment_name="test",
            _output_root=str(tmp_path),
            transparency={"test_explainer": transparency_config},
        ),
    )


@pytest.mark.usefixtures("needs_captum")
def test_explanation_returns_explanation_result(
    simple_cnn: torch.nn.Module,
    sample_images: torch.Tensor,
    tmp_path: Path,
) -> None:
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "_target_": "raitap.transparency.CaptumExplainer",
                "algorithm": "Saliency",
                "call": {"target": 0},
                "visualisers": [{"_target_": "raitap.transparency.CaptumImageVisualiser"}],
            }
        ),
    )

    model = SimpleNamespace(backend=_BackendStub(simple_cnn))
    explanation = Explanation(config, "test_explainer", model, sample_images)  # type: ignore[arg-type]

    assert isinstance(explanation, ExplanationResult)
    assert explanation.attributions.shape == sample_images.shape
    assert explanation.run_dir == tmp_path / "transparency" / "test_explainer"
    assert (explanation.run_dir / "attributions.pt").exists()
    assert (explanation.run_dir / "metadata.json").exists()

    visualisations = explanation.visualise()
    assert len(visualisations) == 1
    assert (explanation.run_dir / "CaptumImageVisualiser_0.png").exists()


def test_explanation_rejects_model_without_backend(
    sample_images: torch.Tensor,
    tmp_path: Path,
) -> None:
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "_target_": "raitap.transparency.CaptumExplainer",
                "algorithm": "Saliency",
                "call": {"target": 0},
                "visualisers": [],
            }
        ),
    )

    with pytest.raises(TypeError, match=r"ModelBackend instance"):
        Explanation(
            config,
            "test_explainer",
            model=torch.nn.Identity(),  # type: ignore[arg-type]
            inputs=sample_images,
        )


def test_explanation_validates_visualisers_before_compute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class DummyExplainer:
        algorithm = "KernelExplainer"

        def __init__(self) -> None:
            self.explain_called = False

        def check_backend_compat(self, backend: object) -> None:
            del backend
            return None

        def explain(self, *args: Any, **kwargs: Any) -> None:
            self.explain_called = True
            raise AssertionError("explain() should not be called for incompatible visualisers")

    dummy_explainer = DummyExplainer()
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "_target_": "raitap.transparency.ShapExplainer",
                "algorithm": "KernelExplainer",
                "visualisers": [{"_target_": "raitap.transparency.ShapImageVisualiser"}],
            }
        ),
    )

    monkeypatch.setattr(
        "raitap.transparency.factory.create_explainer",
        lambda _config: (dummy_explainer, "raitap.transparency.ShapExplainer"),
    )

    with pytest.raises(VisualiserIncompatibilityError):
        model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
        Explanation(
            config,
            "test_explainer",
            model=model,  # type: ignore[arg-type]
            inputs=torch.zeros(1, 3, 8, 8),
        )

    assert dummy_explainer.explain_called is False


@pytest.mark.usefixtures("needs_captum")
def test_payload_kind_wildcard_visualiser_passes(
    monkeypatch: pytest.MonkeyPatch,
    simple_cnn: torch.nn.Module,
    sample_images: torch.Tensor,
    tmp_path: Path,
) -> None:
    class _WildcardPayloadVisualiser(BaseVisualiser):
        compatible_algorithms: ClassVar[frozenset[str]] = frozenset()
        supported_payload_kinds: ClassVar[frozenset[ExplanationPayloadKind]] = frozenset()

        def visualise(
            self,
            attributions: torch.Tensor,
            inputs: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> Figure:
            del attributions, inputs, kwargs
            fig, _ax = plt.subplots(figsize=(1, 1))
            return fig

    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "_target_": "raitap.transparency.CaptumExplainer",
                "algorithm": "Saliency",
                "call": {"target": 0},
                "visualisers": [{"_target_": "raitap.transparency.CaptumImageVisualiser"}],
            }
        ),
    )
    monkeypatch.setattr(
        "raitap.transparency.factory.create_visualisers",
        lambda _cfg: [
            ConfiguredVisualiser(visualiser=_WildcardPayloadVisualiser(), call_kwargs={})
        ],
    )
    model = SimpleNamespace(backend=_BackendStub(simple_cnn))
    explanation = Explanation(config, "test_explainer", model, sample_images)  # type: ignore[arg-type]

    assert isinstance(explanation, ExplanationResult)


@pytest.mark.usefixtures("needs_captum")
def test_payload_kind_mismatch_raises(
    monkeypatch: pytest.MonkeyPatch,
    simple_cnn: torch.nn.Module,
    sample_images: torch.Tensor,
    tmp_path: Path,
) -> None:
    class _StructuredOnlyVisualiser(BaseVisualiser):
        compatible_algorithms: ClassVar[frozenset[str]] = frozenset()
        supported_payload_kinds: ClassVar[frozenset[ExplanationPayloadKind]] = frozenset(
            {ExplanationPayloadKind.STRUCTURED}
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

    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "_target_": "raitap.transparency.CaptumExplainer",
                "algorithm": "Saliency",
                "call": {"target": 0},
                "visualisers": [{"_target_": "raitap.transparency.CaptumImageVisualiser"}],
            }
        ),
    )
    monkeypatch.setattr(
        "raitap.transparency.factory.create_visualisers",
        lambda _cfg: [ConfiguredVisualiser(visualiser=_StructuredOnlyVisualiser(), call_kwargs={})],
    )
    model = SimpleNamespace(backend=_BackendStub(simple_cnn))
    with pytest.raises(PayloadVisualiserIncompatibilityError):
        Explanation(config, "test_explainer", model, sample_images)  # type: ignore[arg-type]


def test_create_explainer_resolves_short_target_and_strips_visualisers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_cfg: dict[str, Any] = {}

    class _StubExplainer:
        def check_backend_compat(self, backend: object) -> None:
            del backend
            return None

        def explain(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    explainer = _StubExplainer()

    def _fake_instantiate(cfg: dict[str, Any]) -> object:
        captured_cfg.update(cfg)
        return explainer

    config = OmegaConf.create(
        {
            "_target_": "CaptumExplainer",
            "algorithm": "Saliency",
            "visualisers": [{"_target_": "raitap.transparency.CaptumImageVisualiser"}],
        }
    )
    monkeypatch.setattr("raitap.transparency.factory.instantiate", _fake_instantiate)

    created, resolved_target = create_explainer(config)

    assert created is explainer
    assert resolved_target == "raitap.transparency.CaptumExplainer"
    assert captured_cfg["_target_"] == "raitap.transparency.CaptumExplainer"
    assert "visualisers" not in captured_cfg


def test_create_explainer_wraps_instantiation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = OmegaConf.create({"_target_": "NoSuchExplainer"})

    def _raise(_: dict[str, Any]) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("raitap.transparency.factory.instantiate", _raise)
    with pytest.raises(ValueError, match="Could not instantiate explainer"):
        create_explainer(config)


def test_create_explainer_rejects_missing_check_backend_compat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ExplainerWithoutBackendCheck:
        def explain(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    monkeypatch.setattr(
        "raitap.transparency.factory.instantiate",
        lambda _cfg: _ExplainerWithoutBackendCheck(),
    )
    config = OmegaConf.create({"_target_": "CaptumExplainer", "algorithm": "Saliency"})
    with pytest.raises(ValueError, match="check_backend_compat"):
        create_explainer(config)


def test_create_explainer_rejects_unknown_top_level_keys() -> None:
    config = OmegaConf.create(
        {
            "_target_": "CaptumExplainer",
            "algorithm": "Saliency",
            "multiply_by_inputs": True,
            "visualisers": [],
        }
    )
    with pytest.raises(ValueError, match="Unknown transparency explainer config keys"):
        create_explainer(config)


def test_create_explainer_warns_on_unknown_raitap_keys(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _StubExplainer:
        def check_backend_compat(self, backend: object) -> None:
            del backend
            return None

        def explain(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    monkeypatch.setattr(
        "raitap.transparency.factory.instantiate",
        lambda _cfg: _StubExplainer(),
    )
    config = OmegaConf.create(
        {
            "_target_": "CaptumExplainer",
            "algorithm": "Saliency",
            "raitap": {"bacth_size": 2},
        }
    )

    with caplog.at_level("WARNING"):
        create_explainer(config)

    assert "bacth_size" in caplog.text


def test_create_explainer_warns_on_misplaced_raitap_call_keys(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _StubExplainer:
        def check_backend_compat(self, backend: object) -> None:
            del backend
            return None

        def explain(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    monkeypatch.setattr(
        "raitap.transparency.factory.instantiate",
        lambda _cfg: _StubExplainer(),
    )
    config = OmegaConf.create(
        {
            "_target_": "CaptumExplainer",
            "algorithm": "Saliency",
            "call": {
                "batch_size": 2,
                "progress_desc": "batches",
                "sample_names": ["a", "b"],
                "show_sample_names": True,
            },
        }
    )

    with caplog.at_level("WARNING"):
        create_explainer(config)

    assert "RAITAP-owned keys under 'call:'" in caplog.text
    assert "batch_size" in caplog.text
    assert "progress_desc" in caplog.text
    assert "sample_names" in caplog.text
    assert "show_sample_names" in caplog.text


def test_create_explainer_does_not_warn_on_call_show_progress(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _StubExplainer:
        def check_backend_compat(self, backend: object) -> None:
            del backend
            return None

        def explain(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    monkeypatch.setattr(
        "raitap.transparency.factory.instantiate",
        lambda _cfg: _StubExplainer(),
    )
    config = OmegaConf.create(
        {
            "_target_": "CaptumExplainer",
            "algorithm": "KernelShap",
            "call": {"show_progress": True},
        }
    )

    with caplog.at_level("WARNING"):
        create_explainer(config)

    assert "RAITAP-owned keys under 'call:'" not in caplog.text


def test_create_explainer_rejects_removed_max_batch_size_raitap_key() -> None:
    config = OmegaConf.create(
        {
            "_target_": "CaptumExplainer",
            "algorithm": "Saliency",
            "raitap": {"max_batch_size": 2},
        }
    )

    with pytest.raises(
        ValueError,
        match=r"raitap\.max_batch_size has been removed; use raitap\.batch_size instead\.",
    ):
        create_explainer(config)


def test_create_explainer_forwards_constructor_to_instantiate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_cfg: dict[str, Any] = {}

    class _StubExplainer2:
        def check_backend_compat(self, backend: object) -> None:
            del backend
            return None

        def explain(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    def _fake_instantiate(cfg: dict[str, Any]) -> object:
        captured_cfg.update(cfg)
        return _StubExplainer2()

    config = OmegaConf.create(
        {
            "_target_": "CaptumExplainer",
            "algorithm": "Saliency",
            "constructor": {"multiply_by_inputs": True},
            "call": {"target": 0},
            "visualisers": [],
        }
    )
    monkeypatch.setattr("raitap.transparency.factory.instantiate", _fake_instantiate)

    create_explainer(config)

    assert captured_cfg["multiply_by_inputs"] is True
    assert captured_cfg["algorithm"] == "Saliency"
    assert "call" not in captured_cfg
    assert "constructor" not in captured_cfg
    assert "target" not in captured_cfg


def test_create_explainer_and_visualisers_from_real_shap_preset() -> None:
    config = _load_transparency_preset("shap_gradient")

    explainer, resolved_target = create_explainer(config)
    visualisers = create_visualisers(config)

    assert type(explainer).__name__ == "ShapExplainer"
    assert cast("Any", explainer).algorithm == "GradientExplainer"
    assert resolved_target == "raitap.transparency.ShapExplainer"
    assert len(visualisers) == 1
    assert type(visualisers[0].visualiser).__name__ == "ShapImageVisualiser"
    assert visualisers[0].call_kwargs == {}
    assert visualisers[0].visualiser.compatible_algorithms == frozenset(
        {"GradientExplainer", "DeepExplainer"}
    )


def test_explanation_merges_call_before_run_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_images: torch.Tensor,
) -> None:
    """YAML ``call`` supplies defaults; call-site kwargs override the same keys."""

    class RecordingExplainer:
        algorithm = "Saliency"

        def __init__(self) -> None:
            self.last_explain_kwargs: dict[str, Any] = {}

        def check_backend_compat(self, backend: object) -> None:
            del backend
            return None

        def explain(self, *_args: Any, **kwargs: Any) -> ExplanationResult:
            self.last_explain_kwargs = dict(kwargs)
            return MagicMock(spec=ExplanationResult)

    explainer = RecordingExplainer()
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "_target_": "raitap.transparency.CaptumExplainer",
                "algorithm": "Saliency",
                "call": {"target": 0, "baselines": "from_yaml"},
                "visualisers": [{"_target_": "raitap.transparency.CaptumImageVisualiser"}],
            }
        ),
    )

    vis = MagicMock()
    vis.compatible_algorithms = frozenset({"Saliency"})

    monkeypatch.setattr(
        "raitap.transparency.factory.create_explainer",
        lambda _cfg: (explainer, "raitap.transparency.CaptumExplainer"),
    )
    monkeypatch.setattr(
        "raitap.transparency.factory.create_visualisers",
        lambda _cfg: [ConfiguredVisualiser(visualiser=vis, call_kwargs={})],
    )

    model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
    Explanation(
        config,
        "test_explainer",
        model=model,  # type: ignore[arg-type]
        inputs=sample_images,
        target=7,
    )

    assert explainer.last_explain_kwargs["target"] == 7
    assert explainer.last_explain_kwargs["baselines"] == "from_yaml"


def test_explanation_uses_real_shap_preset_defaults_and_runtime_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_images: torch.Tensor,
) -> None:
    class RecordingExplainer:
        algorithm = "GradientExplainer"

        def __init__(self) -> None:
            self.last_explain_kwargs: dict[str, Any] = {}

        def check_backend_compat(self, backend: object) -> None:
            del backend
            return None

        def explain(self, *_args: Any, **kwargs: Any) -> ExplanationResult:
            self.last_explain_kwargs = dict(kwargs)
            return MagicMock(spec=ExplanationResult)

    explainer = RecordingExplainer()
    background_data = sample_images[:2]
    config = cast(
        "AppConfig",
        cast(
            "object",
            SimpleNamespace(
                experiment_name="test",
                _output_root=str(tmp_path),
                transparency={"shap_gradient": _load_transparency_preset("shap_gradient")},
            ),
        ),
    )

    vis = MagicMock()
    vis.compatible_algorithms = frozenset({"GradientExplainer", "DeepExplainer"})

    monkeypatch.setattr(
        "raitap.transparency.factory.create_explainer",
        lambda _cfg: (explainer, "raitap.transparency.ShapExplainer"),
    )
    monkeypatch.setattr(
        "raitap.transparency.factory.create_visualisers",
        lambda _cfg: [ConfiguredVisualiser(visualiser=vis, call_kwargs={})],
    )

    model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
    Explanation(
        config,
        "shap_gradient",
        model=model,  # type: ignore[arg-type]
        inputs=sample_images,
        target=7,
        background_data=background_data,
    )

    assert explainer.last_explain_kwargs["target"] == 7
    assert explainer.last_explain_kwargs["background_data"] is background_data
    assert explainer.last_explain_kwargs["nsamples"] == 10
    assert explainer.last_explain_kwargs["raitap_kwargs"] == {
        "batch_size": 1,
        "progress_desc": "SHAP batches",
    }


def test_explanation_injects_runtime_sample_names_into_raitap_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_images: torch.Tensor,
) -> None:
    class RecordingExplainer:
        algorithm = "Saliency"

        def __init__(self) -> None:
            self.last_explain_kwargs: dict[str, Any] = {}

        def check_backend_compat(self, backend: object) -> None:
            del backend
            return None

        def explain(self, *_args: Any, **kwargs: Any) -> ExplanationResult:
            self.last_explain_kwargs = dict(kwargs)
            return MagicMock(spec=ExplanationResult)

    explainer = RecordingExplainer()
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "_target_": "raitap.transparency.CaptumExplainer",
                "algorithm": "Saliency",
                "raitap": {"show_sample_names": True},
                "visualisers": [],
            }
        ),
    )

    monkeypatch.setattr(
        "raitap.transparency.factory.create_explainer",
        lambda _cfg: (explainer, "raitap.transparency.CaptumExplainer"),
    )
    monkeypatch.setattr("raitap.transparency.factory.create_visualisers", lambda _cfg: [])

    model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
    Explanation(
        config,
        "test_explainer",
        model=model,  # type: ignore[arg-type]
        inputs=sample_images,
        sample_names=["isic_1", "isic_2", "isic_3", "isic_4"],
    )

    assert explainer.last_explain_kwargs["raitap_kwargs"] == {
        "show_sample_names": True,
        "sample_names": ["isic_1", "isic_2", "isic_3", "isic_4"],
    }


def test_explanation_runtime_sample_names_override_raitap_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_images: torch.Tensor,
) -> None:
    class RecordingExplainer:
        algorithm = "Saliency"

        def __init__(self) -> None:
            self.last_explain_kwargs: dict[str, Any] = {}

        def check_backend_compat(self, backend: object) -> None:
            del backend
            return None

        def explain(self, *_args: Any, **kwargs: Any) -> ExplanationResult:
            self.last_explain_kwargs = dict(kwargs)
            return MagicMock(spec=ExplanationResult)

    explainer = RecordingExplainer()
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "_target_": "raitap.transparency.CaptumExplainer",
                "algorithm": "Saliency",
                "raitap": {
                    "sample_names": ["from_config_1", "from_config_2"],
                    "show_sample_names": True,
                },
                "visualisers": [],
            }
        ),
    )

    monkeypatch.setattr(
        "raitap.transparency.factory.create_explainer",
        lambda _cfg: (explainer, "raitap.transparency.CaptumExplainer"),
    )
    monkeypatch.setattr("raitap.transparency.factory.create_visualisers", lambda _cfg: [])

    model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
    Explanation(
        config,
        "test_explainer",
        model=model,  # type: ignore[arg-type]
        inputs=sample_images,
        sample_names=["from_runtime_1", "from_runtime_2"],
    )

    assert explainer.last_explain_kwargs["raitap_kwargs"] == {
        "show_sample_names": True,
        "sample_names": ["from_runtime_1", "from_runtime_2"],
    }


def test_explanation_warns_on_unknown_raitap_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_images: torch.Tensor,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class RecordingExplainer:
        algorithm = "Saliency"

        def check_backend_compat(self, backend: object) -> None:
            del backend
            return None

        def explain(self, *_args: Any, **_kwargs: Any) -> ExplanationResult:
            return MagicMock(spec=ExplanationResult)

    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "_target_": "raitap.transparency.CaptumExplainer",
                "algorithm": "Saliency",
                "raitap": {"bacth_size": 2},
                "visualisers": [],
            }
        ),
    )

    monkeypatch.setattr(
        "raitap.transparency.factory.create_explainer",
        lambda _cfg: (RecordingExplainer(), "raitap.transparency.CaptumExplainer"),
    )
    monkeypatch.setattr("raitap.transparency.factory.create_visualisers", lambda _cfg: [])

    model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
    with caplog.at_level("WARNING"):
        Explanation(
            config,
            "test_explainer",
            model=model,  # type: ignore[arg-type]
            inputs=sample_images,
        )

    assert "bacth_size" in caplog.text


def test_explanation_warns_once_on_misplaced_raitap_call_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_images: torch.Tensor,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _StubExplainer:
        algorithm = "Saliency"

        def check_backend_compat(self, backend: object) -> None:
            del backend
            return None

        def explain(self, *_args: Any, **_kwargs: Any) -> ExplanationResult:
            return MagicMock(spec=ExplanationResult)

    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "_target_": "raitap.transparency.CaptumExplainer",
                "algorithm": "Saliency",
                "call": {"sample_names": ["cfg_a", "cfg_b"]},
                "visualisers": [],
            }
        ),
    )

    monkeypatch.setattr("raitap.transparency.factory.instantiate", lambda _cfg: _StubExplainer())
    monkeypatch.setattr("raitap.transparency.factory.create_visualisers", lambda _cfg: [])

    model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
    with caplog.at_level("WARNING"):
        Explanation(
            config,
            "test_explainer",
            model=model,  # type: ignore[arg-type]
            inputs=sample_images,
        )

    messages = [
        record.message
        for record in caplog.records
        if "RAITAP-owned keys under 'call:'" in record.message
    ]
    assert len(messages) == 1
    assert "sample_names" in messages[0]


def test_explanation_rejects_removed_max_batch_size_raitap_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_images: torch.Tensor,
) -> None:
    class RecordingExplainer:
        algorithm = "Saliency"

        def check_backend_compat(self, backend: object) -> None:
            del backend
            return None

        def explain(self, *_args: Any, **_kwargs: Any) -> ExplanationResult:
            return MagicMock(spec=ExplanationResult)

    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "_target_": "raitap.transparency.CaptumExplainer",
                "algorithm": "Saliency",
                "raitap": {"max_batch_size": 2},
                "visualisers": [],
            }
        ),
    )

    monkeypatch.setattr(
        "raitap.transparency.factory.create_explainer",
        lambda _cfg: (RecordingExplainer(), "raitap.transparency.CaptumExplainer"),
    )
    monkeypatch.setattr("raitap.transparency.factory.create_visualisers", lambda _cfg: [])

    model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
    with pytest.raises(
        ValueError,
        match=r"raitap\.max_batch_size has been removed; use raitap\.batch_size instead\.",
    ):
        Explanation(
            config,
            "test_explainer",
            model=model,  # type: ignore[arg-type]
            inputs=sample_images,
        )


def test_explanation_prepares_runtime_tensor_kwargs_with_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_images: torch.Tensor,
) -> None:
    class RecordingExplainer:
        algorithm = "Saliency"

        def __init__(self) -> None:
            self.last_explain_kwargs: dict[str, Any] = {}

        def check_backend_compat(self, backend: object) -> None:
            del backend
            return None

        def explain(self, *_args: Any, **kwargs: Any) -> ExplanationResult:
            self.last_explain_kwargs = dict(kwargs)
            return MagicMock(spec=ExplanationResult)

    class PreparingBackend(_BackendStub):
        def _prepare_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
            prepared = dict(kwargs)
            baselines = prepared.get("baselines")
            if isinstance(baselines, torch.Tensor):
                prepared["baselines"] = baselines.to(torch.device("meta"))
            return prepared

    explainer = RecordingExplainer()
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "_target_": "raitap.transparency.CaptumExplainer",
                "algorithm": "Saliency",
                "visualisers": [],
            }
        ),
    )

    monkeypatch.setattr(
        "raitap.transparency.factory.create_explainer",
        lambda _cfg: (explainer, "raitap.transparency.CaptumExplainer"),
    )
    monkeypatch.setattr("raitap.transparency.factory.create_visualisers", lambda _cfg: [])

    model = SimpleNamespace(backend=PreparingBackend(torch.nn.Identity()))
    Explanation(
        config,
        "test_explainer",
        model=model,  # type: ignore[arg-type]
        inputs=sample_images,
        baselines=sample_images[:1],
    )

    baselines = explainer.last_explain_kwargs["baselines"]
    assert isinstance(baselines, torch.Tensor)
    assert baselines.device.type == "meta"


def test_check_explainer_visualiser_compat_allows_compatible() -> None:
    visualiser = MagicMock()
    visualiser.compatible_algorithms = frozenset({"Saliency"})
    check_explainer_visualiser_compat(
        "raitap.transparency.CaptumExplainer",
        "Saliency",
        [ConfiguredVisualiser(visualiser=visualiser, call_kwargs={})],
    )


def test_captum_explainer_blocks_integrated_gradients_on_non_autograd_backend() -> None:
    from raitap.transparency.explainers.captum_explainer import CaptumExplainer

    explainer = CaptumExplainer("IntegratedGradients")

    with pytest.raises(ExplainerBackendIncompatibilityError):
        explainer.check_backend_compat(
            _BackendStub(torch.nn.Identity(), supports_torch_autograd=False)
        )


def test_shap_explainer_blocks_gradient_explainer_on_non_autograd_backend() -> None:
    from raitap.transparency.explainers.shap_explainer import ShapExplainer

    explainer = ShapExplainer("GradientExplainer")

    with pytest.raises(ExplainerBackendIncompatibilityError):
        explainer.check_backend_compat(
            _BackendStub(torch.nn.Identity(), supports_torch_autograd=False)
        )


def test_captum_explainer_allows_feature_ablation_on_non_autograd_backend() -> None:
    from raitap.transparency.explainers.captum_explainer import CaptumExplainer

    explainer = CaptumExplainer("FeatureAblation")

    explainer.check_backend_compat(_BackendStub(torch.nn.Identity(), supports_torch_autograd=False))


def test_shap_explainer_allows_kernel_explainer_on_non_autograd_backend() -> None:
    from raitap.transparency.explainers.shap_explainer import ShapExplainer

    explainer = ShapExplainer("KernelExplainer")

    explainer.check_backend_compat(_BackendStub(torch.nn.Identity(), supports_torch_autograd=False))


def test_create_visualisers_rejects_unknown_keys() -> None:
    config = OmegaConf.create(
        {
            "visualisers": [
                {
                    "_target_": "raitap.transparency.CaptumImageVisualiser",
                    "method": "heat_map",
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="Unknown keys in visualiser config"):
        create_visualisers(config)


def test_create_visualisers_splits_constructor_and_call(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_vis: dict[str, Any] = {}

    def _fake_instantiate(cfg: dict[str, Any]) -> object:
        captured_vis.update(cfg)
        m = MagicMock()
        m.compatible_algorithms = frozenset()
        return m

    config = OmegaConf.create(
        {
            "visualisers": [
                {
                    "_target_": "raitap.transparency.CaptumImageVisualiser",
                    "constructor": {"method": "heat_map", "sign": "positive"},
                    "call": {"max_samples": 3},
                }
            ],
        }
    )
    monkeypatch.setattr("raitap.transparency.factory.instantiate", _fake_instantiate)

    configured = create_visualisers(config)

    assert len(configured) == 1
    assert configured[0].call_kwargs == {"max_samples": 3}
    assert captured_vis["method"] == "heat_map"
    assert captured_vis["sign"] == "positive"
    assert "call" not in captured_vis
    assert "constructor" not in captured_vis


# ---------------------------------------------------------------------------
# _resolve_call_data_sources
# ---------------------------------------------------------------------------


class TestResolveCallDataSources:
    def test_passthrough_when_no_data_references(self) -> None:
        kwargs = {"target": 0, "nsamples": 10}
        assert _resolve_call_data_sources(kwargs) == kwargs

    def test_passthrough_when_value_is_not_a_dict(self) -> None:
        kwargs = {"background_data": None, "target": 1}
        assert _resolve_call_data_sources(kwargs) == kwargs

    def test_passthrough_dict_without_source_key(self) -> None:
        kwargs = {"options": {"n_samples": 5}}
        assert _resolve_call_data_sources(kwargs) == kwargs

    def test_loads_tensor_from_source(self, tmp_path: Path) -> None:
        import numpy as np
        from PIL import Image

        img_dir = tmp_path / "bg"
        img_dir.mkdir()
        for i in range(4):
            arr = np.zeros((8, 8, 3), dtype=np.uint8)
            Image.fromarray(arr, "RGB").save(img_dir / f"img{i}.png")

        result = _resolve_call_data_sources({"background_data": {"source": str(img_dir)}})

        assert isinstance(result["background_data"], torch.Tensor)
        assert result["background_data"].shape == (4, 3, 8, 8)

    def test_n_samples_subsamples(self, tmp_path: Path) -> None:
        import numpy as np
        from PIL import Image

        img_dir = tmp_path / "bg"
        img_dir.mkdir()
        for i in range(10):
            arr = np.zeros((8, 8, 3), dtype=np.uint8)
            Image.fromarray(arr, "RGB").save(img_dir / f"img{i}.png")

        result = _resolve_call_data_sources(
            {"background_data": {"source": str(img_dir), "n_samples": 3}}
        )

        assert result["background_data"].shape[0] == 3

    def test_invalid_n_samples_type_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="n_samples must be an int"):
            _resolve_call_data_sources(
                {"background_data": {"source": str(tmp_path), "n_samples": "bad"}}
            )

    def test_non_data_source_dict_with_extra_keys_is_passed_through(self) -> None:
        """A dict with keys beyond {source, n_samples} is not treated as a data source."""
        value = {"source": "somewhere", "extra_key": True}
        result = _resolve_call_data_sources({"some_kwarg": value})
        assert result["some_kwarg"] is value

    def test_explanation_injects_background_data_from_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        sample_images: torch.Tensor,
    ) -> None:
        """background_data under call: with source notation is resolved and forwarded."""

        class RecordingExplainer:
            algorithm = "GradientExplainer"

            def __init__(self) -> None:
                self.last_explain_kwargs: dict[str, Any] = {}

            def check_backend_compat(self, backend: object) -> None:
                del backend
                return None

            def explain(self, *_args: Any, **kwargs: Any) -> ExplanationResult:
                self.last_explain_kwargs = dict(kwargs)
                return MagicMock(spec=ExplanationResult)

        import numpy as np
        from PIL import Image

        bg_dir = tmp_path / "bg"
        bg_dir.mkdir()
        for i in range(2):
            arr = np.zeros((8, 8, 3), dtype=np.uint8)
            Image.fromarray(arr, "RGB").save(bg_dir / f"bg{i}.png")

        explainer = RecordingExplainer()
        config = _make_config(
            tmp_path,
            OmegaConf.create(
                {
                    "_target_": "raitap.transparency.ShapExplainer",
                    "algorithm": "GradientExplainer",
                    "call": {
                        "target": 0,
                        "background_data": {"source": str(bg_dir)},
                    },
                    "visualisers": [],
                }
            ),
        )

        monkeypatch.setattr(
            "raitap.transparency.factory.create_explainer",
            lambda _cfg: (explainer, "raitap.transparency.ShapExplainer"),
        )
        monkeypatch.setattr(
            "raitap.transparency.factory.create_visualisers",
            lambda _cfg: [],
        )

        model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
        Explanation(config, "test_explainer", model=model, inputs=sample_images)  # type: ignore[arg-type]

        assert "background_data" in explainer.last_explain_kwargs
        bg_tensor = explainer.last_explain_kwargs["background_data"]
        assert isinstance(bg_tensor, torch.Tensor)
        assert bg_tensor.shape[0] == 2
