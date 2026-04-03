"""
End-to-end SHAP tests.

These tests exercise the full pipeline (explainer → attributions → visualiser → saved artefacts)
and are intentionally resource-intensive.  They are tagged ``@pytest.mark.e2e`` and should be
run once per PR rather than on every commit:

    # fast suite (default CI every commit)
    uv run pytest -m "not e2e"

    # E2E suite (once per PR)
    uv run pytest -m e2e
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch
from omegaconf import OmegaConf

from raitap.transparency import ExplanationResult
from raitap.transparency.explainers import ShapExplainer
from raitap.transparency.factory import Explanation
from raitap.transparency.results import ConfiguredVisualiser
from raitap.transparency.visualisers import (
    ShapBarVisualiser,
    ShapBeeswarmVisualiser,
    ShapImageVisualiser,
    ShapWaterfallVisualiser,
)

if TYPE_CHECKING:
    from pathlib import Path

    import torch.nn as nn

    from raitap.configs.schema import AppConfig

# Every test in this module requires shap and carries the e2e marker.
pytestmark = [pytest.mark.e2e, pytest.mark.usefixtures("needs_shap")]


def _read_metadata(run_dir: Path) -> dict[str, object]:
    return cast(
        "dict[str, object]",
        json.loads((run_dir / "metadata.json").read_text(encoding="utf-8")),
    )


def _load_saved_attributions(run_dir: Path) -> torch.Tensor:
    return cast("torch.Tensor", torch.load(run_dir / "attributions.pt"))


# ---------------------------------------------------------------------------
# DeepExplainer — PyTorch-native, moderate cost
# ---------------------------------------------------------------------------


def test_deep_explainer_image_pipeline(
    simple_cnn: nn.Module,
    sample_images: torch.Tensor,
    tmp_path: Path,
) -> None:
    """DeepExplainer + ShapImageVisualiser: full artefact pipeline on a CNN.

    Covers the ``GradientExplainer / DeepExplainer`` tensor branch of
    ``compute_attributions`` and ``ShapImageVisualiser`` end-to-end.
    """
    explainer = ShapExplainer("DeepExplainer")
    background = sample_images[:2]

    explanation = explainer.explain(
        simple_cnn,
        sample_images,
        run_dir=tmp_path / "transparency",
        background_data=background,
        target=0,
        visualisers=[ConfiguredVisualiser(visualiser=ShapImageVisualiser())],
    )
    visualisations = explanation.visualise()
    metadata = _read_metadata(explanation.run_dir)
    saved_attributions = _load_saved_attributions(explanation.run_dir)

    assert isinstance(explanation, ExplanationResult)
    assert explanation.attributions.shape == sample_images.shape
    assert explanation.run_dir == tmp_path / "transparency"
    assert (explanation.run_dir / "attributions.pt").exists()
    assert (explanation.run_dir / "metadata.json").exists()
    assert torch.equal(saved_attributions, explanation.attributions)
    assert metadata["algorithm"] == "DeepExplainer"
    assert str(metadata["target"]).endswith("ShapExplainer")
    assert cast("dict[str, object]", metadata["kwargs"])["target"] == 0
    assert len(cast("list[str]", metadata["visualisers"])) == 1
    assert cast("list[str]", metadata["visualisers"])[0].endswith("ShapImageVisualiser_0")
    assert len(visualisations) == 1
    assert visualisations[0].output_path == explanation.run_dir / "ShapImageVisualiser_0.png"
    assert visualisations[0].output_path.exists()


def test_deep_explainer_multiclass_all_targets(
    simple_cnn: nn.Module,
    sample_images: torch.Tensor,
    tmp_path: Path,
) -> None:
    """DeepExplainer without a target returns one attribution map per output class.

    Covers the ``isinstance(shap_values, list)`` → stack branch in
    ``compute_attributions`` and the no-target shortcut that keeps the extra
    class dimension.
    """
    explainer = ShapExplainer("DeepExplainer")
    background = sample_images[:2]

    # No target → shap_values is a list[array], one per output class
    all_class_attrs = explainer.compute_attributions(
        simple_cnn, sample_images, background_data=background
    )

    assert isinstance(all_class_attrs, torch.Tensor)
    # extra trailing class dimension
    assert all_class_attrs.shape[:-1] == sample_images.shape


# ---------------------------------------------------------------------------
# GradientExplainer — fast gradient-based, tabular data
# ---------------------------------------------------------------------------


def test_gradient_explainer_per_sample_list_targets_beeswarm(
    simple_mlp: nn.Module,
    sample_tabular: torch.Tensor,
    tmp_path: Path,
) -> None:
    """GradientExplainer with per-sample ``list`` targets + ShapBeeswarmVisualiser.

    Covers the ``shap_values[batch_indices, ..., target]`` advanced-indexing
    branch and the full pipeline on tabular data.
    """
    explainer = ShapExplainer("GradientExplainer")
    background = sample_tabular[:4]
    n = sample_tabular.shape[0]  # 8
    targets = [i % 2 for i in range(n)]  # alternating 0/1 per sample

    explanation = explainer.explain(
        simple_mlp,
        sample_tabular,
        run_dir=tmp_path / "transparency",
        background_data=background,
        target=targets,
        visualisers=[
            ConfiguredVisualiser(
                visualiser=ShapBeeswarmVisualiser(
                    feature_names=[f"f{i}" for i in range(sample_tabular.shape[1])]
                )
            )
        ],
    )
    visualisations = explanation.visualise()
    metadata = _read_metadata(explanation.run_dir)
    saved_attributions = _load_saved_attributions(explanation.run_dir)

    assert isinstance(explanation, ExplanationResult)
    assert explanation.attributions.shape == sample_tabular.shape
    assert torch.equal(saved_attributions, explanation.attributions)
    assert metadata["algorithm"] == "GradientExplainer"
    assert cast("dict[str, object]", metadata["kwargs"])["target"] == targets
    assert cast("list[str]", metadata["visualisers"])[0].endswith("ShapBeeswarmVisualiser_0")
    assert len(visualisations) == 1
    assert visualisations[0].output_path == explanation.run_dir / "ShapBeeswarmVisualiser_0.png"
    assert visualisations[0].output_path.exists()


def test_gradient_explainer_tensor_target_indexing(
    simple_mlp: nn.Module,
    sample_tabular: torch.Tensor,
) -> None:
    """GradientExplainer with a ``torch.Tensor`` per-sample target.

    Covers the ``isinstance(target, list)`` → ``torch.tensor(target)`` branch
    and the subsequent advanced indexing path.
    """
    explainer = ShapExplainer("GradientExplainer")
    background = sample_tabular[:4]
    n = sample_tabular.shape[0]
    target_tensor = torch.zeros(n, dtype=torch.long)

    attributions = explainer.compute_attributions(
        simple_mlp,
        sample_tabular,
        background_data=background,
        target=target_tensor,
    )

    assert isinstance(attributions, torch.Tensor)
    assert attributions.shape == sample_tabular.shape


def test_gradient_explainer_waterfall_visualiser(
    simple_mlp: nn.Module,
    sample_tabular: torch.Tensor,
    tmp_path: Path,
) -> None:
    """GradientExplainer + ShapWaterfallVisualiser with explicit expected_value."""
    explainer = ShapExplainer("GradientExplainer")
    background = sample_tabular[:4]

    explanation = explainer.explain(
        simple_mlp,
        sample_tabular,
        run_dir=tmp_path / "transparency",
        background_data=background,
        target=0,
        visualisers=[
            ConfiguredVisualiser(
                visualiser=ShapWaterfallVisualiser(
                    feature_names=[f"f{i}" for i in range(sample_tabular.shape[1])],
                    expected_value=0.5,
                    sample_index=1,
                    max_display=5,
                )
            )
        ],
    )
    visualisations = explanation.visualise()
    metadata = _read_metadata(explanation.run_dir)

    assert len(visualisations) == 1
    assert metadata["algorithm"] == "GradientExplainer"
    assert cast("dict[str, object]", metadata["kwargs"])["target"] == 0
    assert cast("list[str]", metadata["visualisers"])[0].endswith("ShapWaterfallVisualiser_0")
    assert visualisations[0].output_path == explanation.run_dir / "ShapWaterfallVisualiser_0.png"
    assert visualisations[0].output_path.exists()


# ---------------------------------------------------------------------------
# ShapBarVisualiser — end-to-end with GradientExplainer (tabular)
# ---------------------------------------------------------------------------


def test_gradient_explainer_bar_visualiser_with_inputs(
    simple_mlp: nn.Module,
    sample_tabular: torch.Tensor,
    tmp_path: Path,
) -> None:
    """ShapBarVisualiser receives ``inputs`` for feature-value colouring.

    Exercises the ``feats = _to_numpy(inputs)`` branch inside ``ShapBarVisualiser``.
    """
    explainer = ShapExplainer("GradientExplainer")
    background = sample_tabular[:4]

    explanation = explainer.explain(
        simple_mlp,
        sample_tabular,
        run_dir=tmp_path / "transparency",
        background_data=background,
        target=0,
        visualisers=[
            ConfiguredVisualiser(
                visualiser=ShapBarVisualiser(
                    feature_names=[f"f{i}" for i in range(sample_tabular.shape[1])]
                )
            )
        ],
    )
    visualisations = explanation.visualise()
    metadata = _read_metadata(explanation.run_dir)

    assert len(visualisations) == 1
    assert metadata["algorithm"] == "GradientExplainer"
    assert cast("dict[str, object]", metadata["kwargs"])["target"] == 0
    assert cast("list[str]", metadata["visualisers"])[0].endswith("ShapBarVisualiser_0")
    assert visualisations[0].output_path == explanation.run_dir / "ShapBarVisualiser_0.png"
    assert visualisations[0].output_path.exists()


# ---------------------------------------------------------------------------
# Full factory pipeline (Explanation helper, not monkeypatched)
# ---------------------------------------------------------------------------


def test_explanation_factory_shap_gradient_full_pipeline(
    simple_cnn: nn.Module,
    sample_images: torch.Tensor,
    tmp_path: Path,
) -> None:
    """Full ``Explanation(config, ...)`` pipeline with a real ShapExplainer.

    This is the closest analogue to the Captum factory E2E test in
    ``test_e2e_integration.py``.  ``background_data`` is injected at
    call-site because OmegaConf cannot serialise a ``torch.Tensor``.
    """
    config = cast(
        "AppConfig",
        SimpleNamespace(
            experiment_name="test_shap_e2e",
            fallback_output_dir=str(tmp_path),
            transparency={
                "shap_gradient": OmegaConf.create(
                    {
                        "_target_": "raitap.transparency.ShapExplainer",
                        "algorithm": "GradientExplainer",
                        "call": {"target": 0},
                        "visualisers": [{"_target_": "raitap.transparency.ShapImageVisualiser"}],
                    }
                )
            },
        ),
    )

    model = SimpleNamespace(network=simple_cnn)
    # background_data is passed as a run-time kwarg (merged into call kwargs by the factory)
    explanation = Explanation(
        config,
        "shap_gradient",
        model,  # type: ignore[arg-type]
        sample_images,
        background_data=sample_images[:2],
    )

    assert isinstance(explanation, ExplanationResult)
    assert explanation.attributions.shape == sample_images.shape
    assert explanation.run_dir == tmp_path / "transparency" / "shap_gradient"
    assert (explanation.run_dir / "attributions.pt").exists()
    assert (explanation.run_dir / "metadata.json").exists()
    saved_attributions = _load_saved_attributions(explanation.run_dir)
    metadata = _read_metadata(explanation.run_dir)
    assert torch.equal(saved_attributions, explanation.attributions)
    assert metadata["algorithm"] == "GradientExplainer"
    assert str(metadata["target"]).endswith("ShapExplainer")
    assert cast("dict[str, object]", metadata["kwargs"])["target"] == 0

    visualisations = explanation.visualise()
    assert len(visualisations) == 1
    metadata = _read_metadata(explanation.run_dir)
    assert cast("list[str]", metadata["visualisers"])[0].endswith("ShapImageVisualiser_0")
    assert visualisations[0].output_path == explanation.run_dir / "ShapImageVisualiser_0.png"
    assert visualisations[0].output_path.exists()
