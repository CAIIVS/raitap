from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, cast
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import pytest
import torch
from omegaconf import OmegaConf

from raitap.configs.adapter_factory import resolve_call_data_sources
from raitap.configs.schema import DataConfig
from raitap.models.base_backend import ModelBackend
from raitap.transparency import (
    PayloadVisualiserIncompatibilityError,
    VisualiserIncompatibilityError,
)
from raitap.transparency.contracts import (
    ExplainerAlgorithmSpec,
    ExplanationOutputSpace,
    ExplanationPayloadKind,
    InputSpec,
    MethodFamily,
)
from raitap.transparency.factory import (
    _PARSED_EXPLAINER_CONFIG_CACHE,
    check_explainer_visualiser_compat,
    create_explainer,
    create_visualisers,
)
from raitap.transparency.phase import prepare_explainer
from raitap.transparency.results import ConfiguredVisualiser, ExplanationResult
from raitap.transparency.visualisers.base_visualiser import BaseVisualiser
from raitap.types import Capability
from raitap.utils.errors import BackendIncompatibilityError, SampleNamesLengthError

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure

    from raitap.configs.schema import AppConfig
    from raitap.models import Model


#: Explainer algorithms that run forward-only (model-agnostic). Everything else
#: in these doubles (Saliency, IntegratedGradients, GradientExplainer, ...) is a
#: gradient method needing the live nn.Module. Mirrors the production registries
#: in ``captum_explainer``/``shap_explainer``.
_MODEL_AGNOSTIC_ALGORITHMS = frozenset({"KernelExplainer", "FeatureAblation"})


def _caps_for(algorithm: str) -> frozenset[Capability]:
    """Capabilities a double's algorithm needs, so ``explanation_model`` routes it
    exactly as the real explainer would (autograd_module vs predict_callable)."""
    if algorithm in _MODEL_AGNOSTIC_ALGORITHMS:
        return frozenset()
    return frozenset({Capability.AUTOGRAD})


class _BackendStub(ModelBackend):
    def __init__(self, model: torch.nn.Module, *, autograd: bool = True) -> None:
        self._model = model
        self.provides = frozenset({Capability.AUTOGRAD}) if autograd else frozenset()  # type: ignore[misc]

    @property
    def hardware_label(self) -> str:
        return "stub"

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def _prepare_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return kwargs

    def __call__(self, inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self._model(inputs)

    def autograd_module(self) -> torch.nn.Module:
        return self._model


def _load_transparency_preset(name: str) -> Any:
    """Return an inline SHAP-gradient transparency config for legacy tests.

    The bundled ``shap_gradient`` preset was removed when shipped configs were
    pared down to ``use:``-only stubs. Tests that previously loaded the
    YAML now consume this inlined equivalent — the YAML structure is
    intentionally preserved so the assertions still match the original
    composition.
    """
    del name
    return {
        "use": "shap",
        "algorithm": "GradientExplainer",
        "constructor": {"local_smoothing": 0.0},
        "call": {"target": 0, "nsamples": 10},
        "raitap": {"batch_size": 1, "progress_desc": "SHAP batches"},
        "visualisers": [
            {
                "use": "shap_image",
                "constructor": {"max_samples": 1},
            },
        ],
    }


def _make_config(tmp_path: Path, transparency_config: Any) -> AppConfig:
    return cast(
        "AppConfig",
        SimpleNamespace(
            experiment_name="test",
            _output_root=str(tmp_path),
            transparency={"test_explainer": transparency_config},
            # ``resolve_per_image_transform``'s no-``resolved_preprocessing``
            # fallback reads ``config.data`` directly; a real (defaulted)
            # ``DataConfig`` keeps that read honest instead of re-adding a
            # ``getattr`` default here.
            data=DataConfig(),
        ),
    )


def _image_input_metadata(inputs: torch.Tensor) -> InputSpec:
    return InputSpec(
        kind="image",
        shape=tuple(inputs.shape),
        layout="NCHW",
        metadata={"kind": "image", "layout": "NCHW"},
    )


def _prepare_explanation(
    config: AppConfig,
    name: str,
    model: Model,
    inputs: torch.Tensor,
    *,
    input_metadata: Any = None,
    sample_ids: list[str] | None = None,
    sample_names: list[str] | None = None,
    resolved_preprocessing: Any = None,
    **call_kwargs: Any,
) -> Any:
    """Drive the production *setup* seam with the old ``Explanation`` ergonomics.

    Rebuilds the merged explainer config (runtime ``**call_kwargs`` folded into
    ``call:``, ``sample_names`` folded into ``raitap.sample_names``), then calls
    :func:`prepare_explainer`. Setup-only tests (compat errors, kwargs/provenance,
    run-dir, sample-names length raised in ``explain``) use this and the returned
    :class:`PreparedExplainer`; build tests pass it on to
    :func:`_build_explanation`.

    ``sample_names`` is routed through ``raitap.sample_names`` (the new path has
    no runtime display-name layer; ``data.sample_ids`` doubles as display names).
    Runtime ``**call_kwargs`` may be raw tensors, so the merged config is a plain
    dict (OmegaConf rejects tensors) — ``create_explainer`` accepts plain dicts.
    """
    from raitap.configs.utils import cfg_to_dict

    raw = cfg_to_dict(config.transparency[name])
    call = dict(raw.get("call") or {})
    call.update(call_kwargs)
    raw["call"] = call
    if sample_names is not None:
        raitap_block = dict(raw.get("raitap") or {})
        raitap_block["sample_names"] = sample_names
        raw["raitap"] = raitap_block
    config.transparency[name] = raw  # type: ignore[index]

    if not hasattr(config, "model"):
        config.model = SimpleNamespace(class_names=None)  # type: ignore[attr-defined]
    elif not hasattr(config.model, "class_names"):
        config.model.class_names = None  # type: ignore[attr-defined]

    data = SimpleNamespace(tensor=inputs, sample_ids=sample_ids)
    return prepare_explainer(
        config,
        name,
        model,
        resolved_preprocessing=resolved_preprocessing,
        input_metadata=input_metadata,
        data=cast("Any", data),
    )


def _build_explanation(
    config: AppConfig,
    name: str,
    model: Model,
    inputs: torch.Tensor,
    *,
    input_metadata: Any = None,
    sample_ids: list[str] | None = None,
    sample_names: list[str] | None = None,
    resolved_preprocessing: Any = None,
    forward_output: Any = None,
    **call_kwargs: Any,
) -> ExplanationResult:
    """Drive the full production seam: ``prepare_explainer`` -> ``explain`` -> ``[0]``.

    Mirrors the old one-call ``Explanation(...)`` ergonomics for build tests
    (returned result, visualisations, sample-names length error, auto_pred
    target). ``forward_output`` defaults to a zeros classification logits batch
    sized to ``inputs`` (``ClassificationFamily.explain`` always unwraps it).
    """
    from raitap.pipeline.outputs import ForwardOutput
    from raitap.task_families.classification import ClassificationFamily
    from raitap.types import TaskKind

    prepared = _prepare_explanation(
        config,
        name,
        model,
        inputs,
        input_metadata=input_metadata,
        sample_ids=sample_ids,
        sample_names=sample_names,
        resolved_preprocessing=resolved_preprocessing,
        **call_kwargs,
    )
    if forward_output is None:
        batch = int(inputs.shape[0])
        forward_output = ForwardOutput(
            task_kind=TaskKind.classification,
            batch_size=batch,
            payload=torch.zeros(batch, 2),
        )
    data = SimpleNamespace(tensor=inputs, sample_ids=sample_ids)
    results = ClassificationFamily().explain(
        cast(
            "Any",
            SimpleNamespace(prepared=prepared, forward_output=forward_output, data=data),
        )
    )
    return cast("ExplanationResult", results[0])


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
                "use": "captum",
                "algorithm": "Saliency",
                "call": {"target": 0},
                "visualisers": [{"use": "captum_image"}],
            }
        ),
    )

    model = cast("Model", SimpleNamespace(backend=_BackendStub(simple_cnn)))
    explanation = _build_explanation(
        config,
        "test_explainer",
        model,
        sample_images,
        input_metadata=_image_input_metadata(sample_images),
        target=0,
    )

    assert isinstance(explanation, ExplanationResult)
    assert explanation.attributions.shape == sample_images.shape
    assert explanation.run_dir == tmp_path / "transparency" / "test_explainer"
    assert (explanation.run_dir / "attributions.pt").exists()
    assert (explanation.run_dir / "metadata.json").exists()

    visualisations = explanation._visualise()
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
                "use": "captum",
                "algorithm": "Saliency",
                "call": {"target": 0},
                "visualisers": [],
            }
        ),
    )

    with pytest.raises(TypeError, match=r"ModelBackend instance"):
        _prepare_explanation(
            config,
            "test_explainer",
            cast("Model", torch.nn.Identity()),
            sample_images,
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

        def required_capabilities(self) -> frozenset[Capability]:
            return _caps_for(self.algorithm)

        def explain(self, *args: Any, **kwargs: Any) -> None:
            self.explain_called = True
            raise AssertionError("explain() should not be called for incompatible visualisers")

    dummy_explainer = DummyExplainer()
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "use": "shap",
                "algorithm": "KernelExplainer",
                "visualisers": [{"use": "shap_image"}],
            }
        ),
    )

    monkeypatch.setattr(
        "raitap.transparency.factory.create_explainer",
        lambda _config: (dummy_explainer, "raitap.transparency.ShapExplainer"),
    )

    with pytest.raises(VisualiserIncompatibilityError):
        model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
        _prepare_explanation(
            config,
            "test_explainer",
            cast("Model", model),
            torch.zeros(1, 3, 8, 8),
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
                "use": "captum",
                "algorithm": "Saliency",
                "call": {"target": 0},
                "visualisers": [{"use": "captum_image"}],
            }
        ),
    )
    monkeypatch.setattr(
        "raitap.transparency.factory.create_visualisers",
        lambda _cfg: [
            ConfiguredVisualiser(visualiser=_WildcardPayloadVisualiser(), call_kwargs={})
        ],
    )
    model = cast("Model", SimpleNamespace(backend=_BackendStub(simple_cnn)))
    explanation = _build_explanation(
        config,
        "test_explainer",
        model,
        sample_images,
        input_metadata=_image_input_metadata(sample_images),
        target=0,
    )

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
                "use": "captum",
                "algorithm": "Saliency",
                "call": {"target": 0},
                "visualisers": [{"use": "captum_image"}],
            }
        ),
    )
    monkeypatch.setattr(
        "raitap.transparency.factory.create_visualisers",
        lambda _cfg: [ConfiguredVisualiser(visualiser=_StructuredOnlyVisualiser(), call_kwargs={})],
    )
    model = SimpleNamespace(backend=_BackendStub(simple_cnn))
    with pytest.raises(PayloadVisualiserIncompatibilityError):
        _prepare_explanation(config, "test_explainer", cast("Model", model), sample_images)


def test_explanation_passes_sample_ids_display_names_and_input_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_images: torch.Tensor,
) -> None:
    class RecordingExplainer:
        algorithm = "Saliency"
        output_payload_kind: ClassVar[ExplanationPayloadKind] = ExplanationPayloadKind.ATTRIBUTIONS

        def __init__(self) -> None:
            self.last_raitap_kwargs: dict[str, Any] = {}

        def check_backend_compat(self, backend: object) -> None:
            del backend

        def required_capabilities(self) -> frozenset[Capability]:
            return _caps_for(self.algorithm)

        def explain(self, *_args: Any, **kwargs: Any) -> ExplanationResult:
            self.last_raitap_kwargs = dict(kwargs["raitap_kwargs"])
            return MagicMock(spec=ExplanationResult)

    explainer = RecordingExplainer()
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "use": "captum",
                "algorithm": "Saliency",
                "raitap": {"sample_names": ["configured-name"], "show_sample_names": True},
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
    # The new path has no distinct runtime display-name layer: ``data.sample_ids``
    # doubles as both stable ids and display names (see ``prepare_explainer``).
    _build_explanation(
        config,
        "test_explainer",
        cast("Model", model),
        sample_images,
        input_metadata={"kind": "image", "shape": sample_images.shape, "layout": "NCHW"},
        sample_ids=["stable-1", "stable-2", "stable-3", "stable-4"],
    )

    assert explainer.last_raitap_kwargs["sample_ids"] == [
        "stable-1",
        "stable-2",
        "stable-3",
        "stable-4",
    ]
    assert explainer.last_raitap_kwargs["sample_names"] == [
        "stable-1",
        "stable-2",
        "stable-3",
        "stable-4",
    ]
    assert explainer.last_raitap_kwargs["input_metadata"]["kind"] == "image"
    assert explainer.last_raitap_kwargs["input_metadata"]["layout"] == "NCHW"
    assert explainer.last_raitap_kwargs["show_sample_names"] is True


def test_explanation_rejects_method_family_mismatch_before_compute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_images: torch.Tensor,
) -> None:
    class GradientExplainer:
        algorithm = "Saliency"
        algorithm_registry: ClassVar[dict[str, ExplainerAlgorithmSpec]] = {
            "Saliency": ExplainerAlgorithmSpec({MethodFamily.GRADIENT})
        }
        output_payload_kind: ClassVar[ExplanationPayloadKind] = ExplanationPayloadKind.ATTRIBUTIONS

        def __init__(self) -> None:
            self.explain_called = False

        def check_backend_compat(self, backend: object) -> None:
            del backend

        def required_capabilities(self) -> frozenset[Capability]:
            return _caps_for(self.algorithm)

        def explain(self, *_args: Any, **_kwargs: Any) -> ExplanationResult:
            self.explain_called = True
            return MagicMock(spec=ExplanationResult)

    class _PerturbationOnlyVisualiser(BaseVisualiser):
        supported_method_families: ClassVar[frozenset[MethodFamily]] = frozenset(
            {MethodFamily.PERTURBATION}
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

    explainer = GradientExplainer()
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "use": "captum",
                "algorithm": "Saliency",
                "visualisers": [{"use": "captum_image"}],
            }
        ),
    )
    monkeypatch.setattr(
        "raitap.transparency.factory.create_explainer",
        lambda _cfg: (explainer, "raitap.transparency.CaptumExplainer"),
    )
    monkeypatch.setattr(
        "raitap.transparency.factory.create_visualisers",
        lambda _cfg: [
            ConfiguredVisualiser(visualiser=_PerturbationOnlyVisualiser(), call_kwargs={})
        ],
    )

    model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
    with pytest.raises(ValueError, match="method families"):
        _prepare_explanation(
            config,
            "test_explainer",
            cast("Model", model),
            sample_images,
            input_metadata={"kind": "image", "layout": "NCHW"},
        )

    assert explainer.explain_called is False


def test_explanation_rejects_candidate_output_space_mismatch_before_compute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_images: torch.Tensor,
) -> None:
    class GradientExplainer:
        algorithm = "Saliency"
        algorithm_registry: ClassVar[dict[str, ExplainerAlgorithmSpec]] = {
            "Saliency": ExplainerAlgorithmSpec({MethodFamily.GRADIENT})
        }
        output_payload_kind: ClassVar[ExplanationPayloadKind] = ExplanationPayloadKind.ATTRIBUTIONS

        def __init__(self) -> None:
            self.explain_called = False

        def check_backend_compat(self, backend: object) -> None:
            del backend

        def required_capabilities(self) -> frozenset[Capability]:
            return _caps_for(self.algorithm)

        def explain(self, *_args: Any, **_kwargs: Any) -> ExplanationResult:
            self.explain_called = True
            return MagicMock(spec=ExplanationResult)

    class _SpatialOnlyVisualiser(BaseVisualiser):
        supported_output_spaces: ClassVar[frozenset[ExplanationOutputSpace]] = frozenset(
            {ExplanationOutputSpace.IMAGE_SPATIAL_MAP}
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

    explainer = GradientExplainer()
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "use": "captum",
                "algorithm": "Saliency",
                "visualisers": [{"use": "captum_image"}],
            }
        ),
    )
    monkeypatch.setattr(
        "raitap.transparency.factory.create_explainer",
        lambda _cfg: (explainer, "raitap.transparency.CaptumExplainer"),
    )
    monkeypatch.setattr(
        "raitap.transparency.factory.create_visualisers",
        lambda _cfg: [ConfiguredVisualiser(visualiser=_SpatialOnlyVisualiser(), call_kwargs={})],
    )

    model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
    with pytest.raises(ValueError, match="candidate output spaces"):
        _prepare_explanation(
            config,
            "test_explainer",
            cast("Model", model),
            sample_images,
            input_metadata={"kind": "image", "layout": "NCHW"},
        )

    assert explainer.explain_called is False


def test_explanation_allows_any_supported_candidate_output_space_before_compute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_images: torch.Tensor,
) -> None:
    class GradientExplainer:
        algorithm = "Saliency"
        algorithm_registry: ClassVar[dict[str, ExplainerAlgorithmSpec]] = {
            "Saliency": ExplainerAlgorithmSpec({MethodFamily.GRADIENT})
        }
        output_payload_kind: ClassVar[ExplanationPayloadKind] = ExplanationPayloadKind.ATTRIBUTIONS

        def __init__(self) -> None:
            self.explain_called = False

        def check_backend_compat(self, backend: object) -> None:
            del backend

        def required_capabilities(self) -> frozenset[Capability]:
            return _caps_for(self.algorithm)

        def explain(self, *_args: Any, **_kwargs: Any) -> ExplanationResult:
            self.explain_called = True
            return MagicMock(spec=ExplanationResult)

    class _InterpretableOnlyVisualiser(BaseVisualiser):
        supported_output_spaces: ClassVar[frozenset[ExplanationOutputSpace]] = frozenset(
            {ExplanationOutputSpace.INTERPRETABLE_FEATURES}
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

    explainer = GradientExplainer()
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "use": "captum",
                "algorithm": "Saliency",
                "visualisers": [{"use": "captum_image"}],
            }
        ),
    )
    monkeypatch.setattr(
        "raitap.transparency.factory.create_explainer",
        lambda _cfg: (explainer, "raitap.transparency.CaptumExplainer"),
    )
    monkeypatch.setattr(
        "raitap.transparency.factory.create_visualisers",
        lambda _cfg: [
            ConfiguredVisualiser(visualiser=_InterpretableOnlyVisualiser(), call_kwargs={})
        ],
    )

    model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
    _build_explanation(
        config,
        "test_explainer",
        cast("Model", model),
        sample_images,
        input_metadata={"kind": "image", "layout": "NCHW"},
    )

    assert explainer.explain_called is True


def test_create_explainer_resolves_use_key_and_strips_visualisers(
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
            "use": "captum",
            "algorithm": "Saliency",
            "visualisers": [{"use": "captum_image"}],
        }
    )
    monkeypatch.setattr("raitap.transparency.factory.instantiate", _fake_instantiate)

    created, resolved_target = create_explainer(config)

    expected_fqn = "raitap.transparency.explainers.captum_explainer.CaptumExplainer"
    assert created is explainer
    assert resolved_target == expected_fqn
    assert captured_cfg["_target_"] == expected_fqn
    assert "visualisers" not in captured_cfg


def test_create_explainer_wraps_instantiation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = OmegaConf.create({"use": "captum"})

    def _raise(_: dict[str, Any]) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("raitap.transparency.factory.instantiate", _raise)
    with pytest.raises(ValueError, match="Could not instantiate explainer"):
        create_explainer(config)


def test_explainer_inherits_check_backend_compat_from_mixin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A configured explainer never lacks check_backend_compat; it is inherited
    # from AdapterMixin. Assert the inherited method is present and callable.
    from raitap._adapters import AdapterMixin

    class _StubExplainerWithInheritance(AdapterMixin):
        def explain(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    stub = _StubExplainerWithInheritance()
    monkeypatch.setattr(
        "raitap.transparency.factory.instantiate",
        lambda _cfg: stub,
    )
    config = OmegaConf.create({"use": "captum", "algorithm": "Saliency"})
    explainer, _ = create_explainer(config)
    assert callable(explainer.check_backend_compat)


def test_create_explainer_rejects_unknown_top_level_keys() -> None:
    config = OmegaConf.create(
        {
            "use": "captum",
            "algorithm": "Saliency",
            "multiply_by_inputs": True,
            "visualisers": [],
        }
    )
    with pytest.raises(ValueError, match="Unknown transparency explainer config keys"):
        create_explainer(config)


def test_create_explainer_rejects_config_target_as_security_surface() -> None:
    """``_target_`` in a config block is arbitrary-callable RCE surface (#301);
    it must never reach ``hydra.utils.instantiate``, even nested under an
    innocuous-looking key like ``constructor``."""
    from raitap.configs.registry_resolve import UnsafeConfigTargetError

    config = OmegaConf.create(
        {
            "use": "captum",
            "algorithm": "Saliency",
            "_target_": "os.system",
        }
    )
    with pytest.raises(UnsafeConfigTargetError, match="_target_"):
        create_explainer(config)


def test_create_explainer_rejects_unknown_use_key() -> None:
    config = OmegaConf.create({"use": "does_not_exist", "algorithm": "Saliency"})
    with pytest.raises(
        ValueError, match=r"Unknown transparency key 'does_not_exist'\. Valid keys:"
    ) as excinfo:
        create_explainer(config)
    # Every registered transparency explainer key must be listed.
    assert "captum" in str(excinfo.value)
    assert "shap" in str(excinfo.value)


def test_create_explainer_warns_on_unknown_raitap_keys(
    monkeypatch: pytest.MonkeyPatch,
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
            "use": "captum",
            "algorithm": "Saliency",
            "raitap": {"bacth_size": 2},
        }
    )

    with pytest.warns(UserWarning, match="bacth_size"):
        create_explainer(config)


def test_create_explainer_rejects_misplaced_raitap_call_keys(
    monkeypatch: pytest.MonkeyPatch,
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
            "use": "captum",
            "algorithm": "Saliency",
            "call": {
                "batch_size": 2,
                "progress_desc": "batches",
                "sample_names": ["a", "b"],
                "show_sample_names": True,
            },
        }
    )

    with pytest.raises(ValueError) as excinfo:
        create_explainer(config)
    text = str(excinfo.value)
    assert "RAITAP-owned keys under 'call:'" in text
    assert "batch_size" in text
    assert "progress_desc" in text
    assert "sample_names" in text
    assert "show_sample_names" in text


def test_create_explainer_rejects_call_show_progress(
    monkeypatch: pytest.MonkeyPatch,
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
            "use": "captum",
            "algorithm": "KernelShap",
            "call": {"show_progress": True},
        }
    )

    with pytest.raises(ValueError) as excinfo:
        create_explainer(config)
    text = str(excinfo.value)
    assert "RAITAP-owned keys under 'call:'" in text
    assert "show_progress" in text


def test_create_explainer_rejects_removed_max_batch_size_raitap_key() -> None:
    config = OmegaConf.create(
        {
            "use": "captum",
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
            "use": "captum",
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
    assert resolved_target == "raitap.transparency.explainers.shap_explainer.ShapExplainer"
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

        def required_capabilities(self) -> frozenset[Capability]:
            return _caps_for(self.algorithm)

        def explain(self, *_args: Any, **kwargs: Any) -> ExplanationResult:
            self.last_explain_kwargs = dict(kwargs)
            return MagicMock(spec=ExplanationResult)

    explainer = RecordingExplainer()
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "use": "captum",
                "algorithm": "Saliency",
                "call": {"target": 0, "baselines": "from_yaml"},
                "visualisers": [{"use": "captum_image"}],
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
    # Runtime ``target=7`` is folded into the explainer config's ``call:`` block
    # by the helper, overriding the YAML default of 0; ``baselines`` keeps its
    # YAML value.
    _build_explanation(
        config,
        "test_explainer",
        cast("Model", model),
        sample_images,
        target=7,
    )

    assert explainer.last_explain_kwargs["target"] == 7
    assert explainer.last_explain_kwargs["baselines"] == "from_yaml"


def test_explanation_routes_raitap_baseline_to_adapter_kwarg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_images: torch.Tensor,
) -> None:
    """``raitap.baseline`` is routed to the adapter's own call kwarg end-to-end."""

    class RecordingExplainer:
        algorithm = "IntegratedGradients"
        baseline_kwarg_name = "baselines"

        def __init__(self) -> None:
            self.last_explain_kwargs: dict[str, Any] = {}

        def check_backend_compat(self, backend: object) -> None:
            del backend
            return None

        def required_capabilities(self) -> frozenset[Capability]:
            return _caps_for(self.algorithm)

        def explain(self, *_args: Any, **kwargs: Any) -> ExplanationResult:
            self.last_explain_kwargs = dict(kwargs)
            return MagicMock(spec=ExplanationResult)

    explainer = RecordingExplainer()
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "use": "captum",
                "algorithm": "IntegratedGradients",
                "call": {"target": 0},
                "raitap": {"baseline": "ROUTED"},
                "visualisers": [{"use": "captum_image"}],
            }
        ),
    )

    vis = MagicMock()
    vis.compatible_algorithms = frozenset({"IntegratedGradients"})

    monkeypatch.setattr(
        "raitap.transparency.factory.create_explainer",
        lambda _cfg: (explainer, "raitap.transparency.CaptumExplainer"),
    )
    monkeypatch.setattr(
        "raitap.transparency.factory.create_visualisers",
        lambda _cfg: [ConfiguredVisualiser(visualiser=vis, call_kwargs={})],
    )

    model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
    _build_explanation(
        config,
        "test_explainer",
        cast("Model", model),
        sample_images,
        target=0,
    )

    # Routed under the adapter's own kwarg, and popped from the raitap block.
    assert explainer.last_explain_kwargs["baselines"] == "ROUTED"
    assert "baseline" not in explainer.last_explain_kwargs["raitap_kwargs"]


def test_raitap_baseline_wins_over_runtime_kwarg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_images: torch.Tensor,
) -> None:
    """A runtime baseline kwarg must not silently override ``raitap.baseline``."""

    class RecordingExplainer:
        algorithm = "IntegratedGradients"
        baseline_kwarg_name = "baselines"

        def __init__(self) -> None:
            self.last_explain_kwargs: dict[str, Any] = {}

        def check_backend_compat(self, backend: object) -> None:
            del backend
            return None

        def required_capabilities(self) -> frozenset[Capability]:
            return _caps_for(self.algorithm)

        def explain(self, *_args: Any, **kwargs: Any) -> ExplanationResult:
            self.last_explain_kwargs = dict(kwargs)
            return MagicMock(spec=ExplanationResult)

    explainer = RecordingExplainer()
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "use": "captum",
                "algorithm": "IntegratedGradients",
                "call": {"target": 0},
                "raitap": {"baseline": "ROUTED"},
                "visualisers": [{"use": "captum_image"}],
            }
        ),
    )

    vis = MagicMock()
    vis.compatible_algorithms = frozenset({"IntegratedGradients"})

    monkeypatch.setattr(
        "raitap.transparency.factory.create_explainer",
        lambda _cfg: (explainer, "raitap.transparency.CaptumExplainer"),
    )
    monkeypatch.setattr(
        "raitap.transparency.factory.create_visualisers",
        lambda _cfg: [ConfiguredVisualiser(visualiser=vis, call_kwargs={})],
    )

    model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
    _build_explanation(
        config,
        "test_explainer",
        cast("Model", model),
        sample_images,
        target=0,
        baselines="RUNTIME",  # runtime override folded into call: — same kwarg
    )

    # raitap.baseline wins over the runtime kwarg (and a warning is logged).
    assert explainer.last_explain_kwargs["baselines"] == "ROUTED"


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

        def required_capabilities(self) -> frozenset[Capability]:
            return _caps_for(self.algorithm)

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
                data=DataConfig(),
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
    _build_explanation(
        config,
        "shap_gradient",
        cast("Model", model),
        sample_images,
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

        def required_capabilities(self) -> frozenset[Capability]:
            return _caps_for(self.algorithm)

        def explain(self, *_args: Any, **kwargs: Any) -> ExplanationResult:
            self.last_explain_kwargs = dict(kwargs)
            return MagicMock(spec=ExplanationResult)

    explainer = RecordingExplainer()
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "use": "captum",
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
    _build_explanation(
        config,
        "test_explainer",
        cast("Model", model),
        sample_images,
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

        def required_capabilities(self) -> frozenset[Capability]:
            return _caps_for(self.algorithm)

        def explain(self, *_args: Any, **kwargs: Any) -> ExplanationResult:
            self.last_explain_kwargs = dict(kwargs)
            return MagicMock(spec=ExplanationResult)

    explainer = RecordingExplainer()
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "use": "captum",
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
    # Runtime sample_names (folded into raitap.sample_names by the helper) win
    # over the config block. The old per-name override debug log is gone — the
    # new path overwrites silently.
    _build_explanation(
        config,
        "test_explainer",
        cast("Model", model),
        sample_images,
        sample_names=["from_runtime_1", "from_runtime_2", "from_runtime_3", "from_runtime_4"],
    )

    assert explainer.last_explain_kwargs["raitap_kwargs"] == {
        "show_sample_names": True,
        "sample_names": ["from_runtime_1", "from_runtime_2", "from_runtime_3", "from_runtime_4"],
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

        def required_capabilities(self) -> frozenset[Capability]:
            return _caps_for(self.algorithm)

        def explain(self, *_args: Any, **_kwargs: Any) -> ExplanationResult:
            return MagicMock(spec=ExplanationResult)

    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "use": "captum",
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
    with pytest.warns(UserWarning, match="bacth_size"):
        _prepare_explanation(
            config,
            "test_explainer",
            cast("Model", model),
            sample_images,
        )


def test_explanation_rejects_misplaced_raitap_call_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_images: torch.Tensor,
) -> None:
    class _StubExplainer:
        algorithm = "Saliency"

        def check_backend_compat(self, backend: object) -> None:
            del backend
            return None

        def required_capabilities(self) -> frozenset[Capability]:
            return _caps_for(self.algorithm)

        def explain(self, *_args: Any, **_kwargs: Any) -> ExplanationResult:
            return MagicMock(spec=ExplanationResult)

    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "use": "captum",
                "algorithm": "Saliency",
                "call": {"sample_names": ["cfg_a", "cfg_b", "cfg_c", "cfg_d"]},
                "visualisers": [],
            }
        ),
    )

    monkeypatch.setattr(
        "raitap.transparency.factory.create_explainer",
        lambda _cfg: (_StubExplainer(), "raitap.transparency.ShapExplainer"),
    )
    monkeypatch.setattr("raitap.transparency.factory.create_visualisers", lambda _cfg: [])

    model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
    with pytest.raises(ValueError) as excinfo:
        _prepare_explanation(
            config,
            "test_explainer",
            cast("Model", model),
            sample_images,
        )

    text = str(excinfo.value)
    assert "RAITAP-owned keys under 'call:'" in text
    assert "sample_names" in text


def test_explanation_clears_parsed_config_cache_on_visualiser_compat_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _StubExplainer:
        algorithm = "KernelExplainer"

        def check_backend_compat(self, backend: object) -> None:
            del backend
            return None

        def required_capabilities(self) -> frozenset[Capability]:
            return _caps_for(self.algorithm)

        def explain(self, *_args: Any, **_kwargs: Any) -> ExplanationResult:
            raise AssertionError("explain() should not be called on incompatibility")

    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "use": "shap",
                "algorithm": "KernelExplainer",
                "raitap": {"sample_names": ["cfg_a", "cfg_b"]},
                "visualisers": [{"use": "shap_image"}],
            }
        ),
    )

    monkeypatch.setattr(
        "raitap.transparency.factory.create_explainer",
        lambda _cfg: (_StubExplainer(), "raitap.transparency.ShapExplainer"),
    )

    model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
    with pytest.raises(VisualiserIncompatibilityError):
        _prepare_explanation(
            config,
            "test_explainer",
            cast("Model", model),
            torch.zeros(1, 3, 8, 8),
        )

    assert _PARSED_EXPLAINER_CONFIG_CACHE == {}


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

        def required_capabilities(self) -> frozenset[Capability]:
            return _caps_for(self.algorithm)

        def explain(self, *_args: Any, **_kwargs: Any) -> ExplanationResult:
            return MagicMock(spec=ExplanationResult)

    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "use": "captum",
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
        _prepare_explanation(
            config,
            "test_explainer",
            cast("Model", model),
            sample_images,
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

        def required_capabilities(self) -> frozenset[Capability]:
            return _caps_for(self.algorithm)

        def explain(self, *_args: Any, **kwargs: Any) -> ExplanationResult:
            self.last_explain_kwargs = dict(kwargs)
            return MagicMock(spec=ExplanationResult)

    class PreparingBackend(_BackendStub):
        def _prepare_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
            prepared = dict(kwargs)
            baselines = prepared.get("baselines")
            if isinstance(baselines, torch.Tensor):
                prepared["baselines"] = cast("Any", baselines).to(torch.device("meta"))
            return prepared

    explainer = RecordingExplainer()
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "use": "captum",
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
    _build_explanation(
        config,
        "test_explainer",
        cast("Model", model),
        sample_images,
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

    with pytest.raises(BackendIncompatibilityError):
        explainer.check_backend_compat(_BackendStub(torch.nn.Identity(), autograd=False))


def test_shap_explainer_blocks_gradient_explainer_on_non_autograd_backend() -> None:
    from raitap.transparency.explainers.shap_explainer import ShapExplainer

    explainer = ShapExplainer("GradientExplainer")

    with pytest.raises(BackendIncompatibilityError):
        explainer.check_backend_compat(_BackendStub(torch.nn.Identity(), autograd=False))


def test_captum_explainer_allows_feature_ablation_on_non_autograd_backend() -> None:
    from raitap.transparency.explainers.captum_explainer import CaptumExplainer

    explainer = CaptumExplainer("FeatureAblation")

    explainer.check_backend_compat(_BackendStub(torch.nn.Identity(), autograd=False))


def test_shap_explainer_allows_kernel_explainer_on_non_autograd_backend() -> None:
    from raitap.transparency.explainers.shap_explainer import ShapExplainer

    explainer = ShapExplainer("KernelExplainer")

    explainer.check_backend_compat(_BackendStub(torch.nn.Identity(), autograd=False))


def test_create_visualisers_accepts_programmatic_builder_shape() -> None:
    """A visualiser built via the registered builder function (e.g.
    ``raitap.transparency.captum_image(constructor={"method": "heat_map"})``)
    round-trips through OmegaConf as a plain ``{use, constructor, call, raitap}``
    dict — the *only* shape ``instantiate_visualisers`` needs to understand.
    There is no separate ``hydra_zen.funcs.zen_processing`` wrapper shape any
    more (Phase B replaced hydra-zen ``builds()`` visualiser builders with the
    plain ``_VisualiserUseBase`` dataclass).
    """
    from raitap.transparency import captum_image

    config = OmegaConf.create(
        {"visualisers": [captum_image(constructor={"method": "heat_map"})]},
    )
    visualisers = create_visualisers(config)
    assert len(visualisers) == 1
    assert visualisers[0].visualiser.method == "heat_map"  # pyright: ignore[reportAttributeAccessIssue]


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
                    "use": "captum_image",
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


def test_create_visualisers_rejects_config_target_as_security_surface() -> None:
    from raitap.configs.registry_resolve import UnsafeConfigTargetError

    config = OmegaConf.create(
        {"visualisers": [{"use": "captum_image", "_target_": "os.system"}]},
    )
    with pytest.raises(UnsafeConfigTargetError, match="_target_"):
        create_visualisers(config)


def test_create_visualisers_rejects_unknown_use_key() -> None:
    config = OmegaConf.create({"visualisers": [{"use": "does_not_exist"}]})
    with pytest.raises(ValueError, match=r"Unknown visualiser key 'does_not_exist'\. Valid keys:"):
        create_visualisers(config)


# ---------------------------------------------------------------------------
# resolve_call_data_sources
# ---------------------------------------------------------------------------


class TestResolveCallDataSources:
    def test_passthrough_when_no_data_references(self) -> None:
        kwargs = {"target": 0, "nsamples": 10}
        assert resolve_call_data_sources(kwargs, log_label="call") == kwargs

    def test_passthrough_when_value_is_not_a_dict(self) -> None:
        kwargs = {"background_data": None, "target": 1}
        assert resolve_call_data_sources(kwargs, log_label="call") == kwargs

    def test_passthrough_dict_without_source_key(self) -> None:
        kwargs = {"options": {"n_samples": 5}}
        assert resolve_call_data_sources(kwargs, log_label="call") == kwargs

    def test_loads_tensor_from_source(self, tmp_path: Path) -> None:
        import numpy as np
        from PIL import Image

        img_dir = tmp_path / "bg"
        img_dir.mkdir()
        for i in range(4):
            arr = np.zeros((8, 8, 3), dtype=np.uint8)
            Image.fromarray(arr, "RGB").save(img_dir / f"img{i}.png")

        result = resolve_call_data_sources(
            {"background_data": {"source": str(img_dir)}}, log_label="call"
        )

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

        result = resolve_call_data_sources(
            {"background_data": {"source": str(img_dir), "n_samples": 3}}, log_label="call"
        )

        assert result["background_data"].shape[0] == 3

    def test_invalid_n_samples_type_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="n_samples must be an int"):
            resolve_call_data_sources(
                {"background_data": {"source": str(tmp_path), "n_samples": "bad"}},
                log_label="call",
            )

    def test_non_data_source_dict_with_extra_keys_is_passed_through(self) -> None:
        """A dict with keys beyond {source, n_samples} is not treated as a data source."""
        value = {"source": "somewhere", "extra_key": True}
        result = resolve_call_data_sources({"some_kwarg": value}, log_label="call")
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

            def required_capabilities(self) -> frozenset[Capability]:
                return _caps_for(self.algorithm)

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
                    "use": "shap",
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
        _build_explanation(config, "test_explainer", cast("Model", model), sample_images)

        assert "background_data" in explainer.last_explain_kwargs
        bg_tensor = explainer.last_explain_kwargs["background_data"]
        assert isinstance(bg_tensor, torch.Tensor)
        assert bg_tensor.shape[0] == 2

    def test_explanation_uses_model_resolved_preprocessing_for_call_data(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        sample_images: torch.Tensor,
    ) -> None:
        """Auxiliary call-data should reuse the run-level preprocessing object."""

        class _ShapeModule(torch.nn.Module):
            def forward(self, image: torch.Tensor) -> torch.Tensor:
                return torch.zeros(3, 5, 5, dtype=image.dtype)

        class RecordingExplainer:
            algorithm = "GradientExplainer"

            def __init__(self) -> None:
                self.last_explain_kwargs: dict[str, Any] = {}

            def check_backend_compat(self, backend: object) -> None:
                del backend

            def required_capabilities(self) -> frozenset[Capability]:
                return _caps_for(self.algorithm)

            def explain(self, *_args: Any, **kwargs: Any) -> ExplanationResult:
                self.last_explain_kwargs = dict(kwargs)
                return MagicMock(spec=ExplanationResult)

        from PIL import Image

        from raitap.data.preprocessing import ResolvedPreprocessing

        bg_dir = tmp_path / "bg"
        bg_dir.mkdir()
        for i in range(2):
            Image.fromarray(torch.zeros(8, 8, 3, dtype=torch.uint8).numpy(), "RGB").save(
                bg_dir / f"bg{i}.png"
            )

        explainer = RecordingExplainer()
        config = _make_config(
            tmp_path,
            OmegaConf.create(
                {
                    "use": "shap",
                    "algorithm": "GradientExplainer",
                    "call": {
                        "target": 0,
                        "background_data": {"source": str(bg_dir)},
                    },
                    "visualisers": [],
                }
            ),
        )
        config.model = SimpleNamespace(source="resnet50")  # type: ignore[attr-defined]
        config.data = SimpleNamespace(preprocessing="model-bundled")  # type: ignore[attr-defined]
        resolved = ResolvedPreprocessing(
            data_module=_ShapeModule(),
            model_module=None,
            data_origin="model-bundled",
            model_origin="off",
            description="supplied",
        )

        monkeypatch.setattr(
            "raitap.transparency.factory.create_explainer",
            lambda _cfg: (explainer, "raitap.transparency.ShapExplainer"),
        )
        monkeypatch.setattr("raitap.transparency.factory.create_visualisers", lambda _cfg: [])
        monkeypatch.setattr(
            "raitap.configs.adapter_factory.resolve_preprocessing",
            MagicMock(side_effect=AssertionError("should not resolve again")),
        )

        model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
        _build_explanation(
            cast("AppConfig", config),
            "test_explainer",
            cast("Model", model),
            sample_images,
            resolved_preprocessing=resolved,
        )

        bg_tensor = explainer.last_explain_kwargs["background_data"]
        assert isinstance(bg_tensor, torch.Tensor)
        assert bg_tensor.shape == (2, 3, 5, 5)

    def test_provenance_out_captures_source_and_n_samples(self, tmp_path: Path) -> None:
        import numpy as np
        from PIL import Image

        img_dir = tmp_path / "bg"
        img_dir.mkdir()
        for i in range(6):
            arr = np.zeros((8, 8, 3), dtype=np.uint8)
            Image.fromarray(arr, "RGB").save(img_dir / f"img{i}.png")

        provenance: dict[str, dict[str, object]] = {}
        result = resolve_call_data_sources(
            {"background_data": {"source": str(img_dir), "n_samples": 3}, "target": 0},
            provenance_out=provenance,
        )

        assert isinstance(result["background_data"], torch.Tensor)
        assert provenance == {"background_data": {"source": str(img_dir), "n_samples": 3}}
        # Non-source kwargs do not appear in provenance.
        assert "target" not in provenance

    def test_provenance_out_unaffected_when_not_passed(self, tmp_path: Path) -> None:
        # Robustness/back-compat callers pass no out-param; behaviour unchanged.
        out = resolve_call_data_sources({"target": 1})
        assert out == {"target": 1}


# ---------------------------------------------------------------------------
# sample_names length validation — Task B1
# ---------------------------------------------------------------------------


def _make_stub_explainer() -> Any:
    class _StubExplainer:
        algorithm = "Saliency"

        def check_backend_compat(self, backend: object) -> None:
            del backend

        def required_capabilities(self) -> frozenset[Capability]:
            return _caps_for(self.algorithm)

        def explain(self, *_args: Any, **_kwargs: Any) -> ExplanationResult:
            return MagicMock(spec=ExplanationResult)

    return _StubExplainer()


def test_explanation_raises_when_runtime_sample_names_longer_than_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stub = _make_stub_explainer()
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "use": "captum",
                "algorithm": "Saliency",
                "visualisers": [],
            }
        ),
    )
    monkeypatch.setattr(
        "raitap.transparency.factory.create_explainer",
        lambda _cfg: (stub, "raitap.transparency.CaptumExplainer"),
    )
    monkeypatch.setattr("raitap.transparency.factory.create_visualisers", lambda _cfg: [])

    inputs = torch.zeros(2, 3, 8, 8)
    model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
    with pytest.raises(SampleNamesLengthError) as info:
        _build_explanation(
            config,
            "test_explainer",
            cast("Model", model),
            inputs,
            sample_names=["a", "b", "c"],
        )
    assert info.value.got == 3
    assert info.value.expected == 2
    # The new path resolves sample_names from raitap.sample_names (no runtime
    # display-name layer), so the error source is always "raitap.sample_names".
    assert "raitap.sample_names" in str(info.value)


def test_explanation_raises_when_runtime_sample_names_shorter_than_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stub = _make_stub_explainer()
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "use": "captum",
                "algorithm": "Saliency",
                "visualisers": [],
            }
        ),
    )
    monkeypatch.setattr(
        "raitap.transparency.factory.create_explainer",
        lambda _cfg: (stub, "raitap.transparency.CaptumExplainer"),
    )
    monkeypatch.setattr("raitap.transparency.factory.create_visualisers", lambda _cfg: [])

    inputs = torch.zeros(3, 3, 8, 8)
    model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
    with pytest.raises(SampleNamesLengthError):
        _build_explanation(
            config,
            "test_explainer",
            cast("Model", model),
            inputs,
            sample_names=["only-one"],
        )


def test_explanation_raises_when_yaml_sample_names_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """YAML raitap.sample_names = ['x', 'y'] but inputs have 3 samples — no runtime kwarg."""
    stub = _make_stub_explainer()
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "use": "captum",
                "algorithm": "Saliency",
                "raitap": {"sample_names": ["x", "y"]},
                "visualisers": [],
            }
        ),
    )
    monkeypatch.setattr(
        "raitap.transparency.factory.create_explainer",
        lambda _cfg: (stub, "raitap.transparency.CaptumExplainer"),
    )
    monkeypatch.setattr("raitap.transparency.factory.create_visualisers", lambda _cfg: [])

    inputs = torch.zeros(3, 3, 8, 8)
    model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
    with pytest.raises(SampleNamesLengthError) as info:
        _build_explanation(
            config,
            "test_explainer",
            cast("Model", model),
            inputs,
        )
    assert "raitap.sample_names" in str(info.value)


def test_explanation_does_not_raise_when_sample_names_is_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stub = _make_stub_explainer()
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "use": "captum",
                "algorithm": "Saliency",
                "visualisers": [],
            }
        ),
    )
    monkeypatch.setattr(
        "raitap.transparency.factory.create_explainer",
        lambda _cfg: (stub, "raitap.transparency.CaptumExplainer"),
    )
    monkeypatch.setattr("raitap.transparency.factory.create_visualisers", lambda _cfg: [])

    inputs = torch.zeros(2, 3, 8, 8)
    model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
    # Should not raise SampleNamesLengthError.
    _build_explanation(
        config,
        "test_explainer",
        cast("Model", model),
        inputs,
        sample_names=None,
    )


def test_explanation_does_not_raise_when_sample_names_length_matches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stub = _make_stub_explainer()
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "use": "captum",
                "algorithm": "Saliency",
                "visualisers": [],
            }
        ),
    )
    monkeypatch.setattr(
        "raitap.transparency.factory.create_explainer",
        lambda _cfg: (stub, "raitap.transparency.CaptumExplainer"),
    )
    monkeypatch.setattr("raitap.transparency.factory.create_visualisers", lambda _cfg: [])

    inputs = torch.zeros(2, 3, 8, 8)
    model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
    # Should not raise.
    _build_explanation(
        config,
        "test_explainer",
        cast("Model", model),
        inputs,
        sample_names=["a", "b"],
    )
