from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from raitap.transparency import VisualiserIncompatibilityError
from raitap.transparency.factory import (
    Explanation,
    _resolve_call_data_sources,
    check_explainer_visualiser_compat,
    create_explainer,
    create_visualisers,
)
from raitap.transparency.results import ConfiguredVisualiser, ExplanationResult

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig


def _make_config(tmp_path: Path, transparency_config: Any) -> AppConfig:
    return cast(
        "AppConfig",
        SimpleNamespace(
            experiment_name="test",
            fallback_output_dir=str(tmp_path),
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

    model = SimpleNamespace(network=simple_cnn)
    explanation = Explanation(config, "test_explainer", model, sample_images)  # type: ignore[arg-type]

    assert isinstance(explanation, ExplanationResult)
    assert explanation.attributions.shape == sample_images.shape
    assert explanation.run_dir == tmp_path / "transparency" / "test_explainer"
    assert (explanation.run_dir / "attributions.pt").exists()
    assert (explanation.run_dir / "metadata.json").exists()

    visualisations = explanation.visualise()
    assert len(visualisations) == 1
    assert (explanation.run_dir / "CaptumImageVisualiser_0.png").exists()


def test_explanation_validates_visualisers_before_compute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class DummyExplainer:
        algorithm = "KernelExplainer"

        def __init__(self) -> None:
            self.explain_called = False

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
        model = SimpleNamespace(network=torch.nn.Identity())
        Explanation(
            config,
            "test_explainer",
            model=model,  # type: ignore[arg-type]
            inputs=torch.zeros(1, 3, 8, 8),
        )

    assert dummy_explainer.explain_called is False


def test_create_explainer_resolves_short_target_and_strips_visualisers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_cfg: dict[str, Any] = {}
    explainer = object()

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


def test_create_explainer_forwards_constructor_to_instantiate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_cfg: dict[str, Any] = {}

    def _fake_instantiate(cfg: dict[str, Any]) -> object:
        captured_cfg.update(cfg)
        return object()

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

    model = SimpleNamespace(network=torch.nn.Identity())
    Explanation(
        config,
        "test_explainer",
        model=model,  # type: ignore[arg-type]
        inputs=sample_images,
        target=7,
    )

    assert explainer.last_explain_kwargs["target"] == 7
    assert explainer.last_explain_kwargs["baselines"] == "from_yaml"


def test_check_explainer_visualiser_compat_allows_compatible() -> None:
    visualiser = MagicMock()
    visualiser.compatible_algorithms = frozenset({"Saliency"})
    check_explainer_visualiser_compat(
        "raitap.transparency.CaptumExplainer",
        "Saliency",
        [ConfiguredVisualiser(visualiser=visualiser, call_kwargs={})],
    )


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

        model = SimpleNamespace(network=torch.nn.Identity())
        Explanation(config, "test_explainer", model=model, inputs=sample_images)  # type: ignore[arg-type]

        assert "background_data" in explainer.last_explain_kwargs
        bg_tensor = explainer.last_explain_kwargs["background_data"]
        assert isinstance(bg_tensor, torch.Tensor)
        assert bg_tensor.shape[0] == 2
