from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

import torch

from raitap.transparency.baselines import (
    _render_baseline_image as _original_render_baseline_image,
)
from raitap.transparency.baselines import (
    build_baseline_record,
)
from raitap.transparency.contracts import InputSpec


def _image_input_spec(shape: tuple[int, ...]) -> InputSpec:
    # InputSpec.__init__ normalises string kind/layout via normalise_* helpers;
    # passing strings avoids depending on enum member names (TensorLayout's NCHW
    # member is BATCH_CHANNEL_HEIGHT_WIDTH, value "NCHW").
    return InputSpec(kind="image", shape=shape, layout="NCHW")


def _tabular_input_spec(shape: tuple[int, ...]) -> InputSpec:
    return InputSpec(kind="tabular", shape=shape, layout="(B,F)")


def _captum_ig() -> SimpleNamespace:
    return SimpleNamespace(
        algorithm="IntegratedGradients",
        baseline_kwarg="baselines",
        baseline_defaults={"IntegratedGradients": "zero"},
    )


def _shap_gradient() -> SimpleNamespace:
    return SimpleNamespace(
        algorithm="GradientExplainer",
        baseline_kwarg="background_data",
        baseline_defaults={"GradientExplainer": "input_batch"},
    )


def test_captum_zero_default_when_baseline_absent(tmp_path: Path) -> None:
    inputs = torch.rand(2, 3, 4, 4)
    record = build_baseline_record(
        explainer=_captum_ig(),
        inputs=inputs,
        call_kwargs={},
        call_provenance=None,
        input_spec=_image_input_spec((2, 3, 4, 4)),
        run_dir=tmp_path,
    )
    assert record is not None
    assert record.mode == "zero"
    assert record.kwarg_name == "baselines"
    assert record.source is None
    assert record.shape == (1, 3, 4, 4)  # torch.zeros_like(inputs[:1])
    expected = build_baseline_record(
        explainer=_captum_ig(),
        inputs=inputs,
        call_kwargs={},
        call_provenance=None,
        input_spec=_image_input_spec((2, 3, 4, 4)),
        run_dir=tmp_path,
    )
    assert expected is not None
    assert record.sha256 == expected.sha256
    assert record.image_path is not None
    assert (tmp_path / record.image_path).exists()


def test_user_tensor_when_baseline_present_without_provenance(tmp_path: Path) -> None:
    inputs = torch.rand(2, 3, 4, 4)
    baseline = torch.rand(1, 3, 4, 4)
    record = build_baseline_record(
        explainer=_captum_ig(),
        inputs=inputs,
        call_kwargs={"baselines": baseline},
        call_provenance=None,
        input_spec=_image_input_spec((2, 3, 4, 4)),
        run_dir=tmp_path,
    )
    assert record is not None
    assert record.mode == "user_tensor"
    assert record.source is None
    assert record.shape == (1, 3, 4, 4)


def test_configured_when_provenance_present(tmp_path: Path) -> None:
    inputs = torch.rand(2, 3, 4, 4)
    background = torch.rand(5, 3, 4, 4)
    record = build_baseline_record(
        explainer=_shap_gradient(),
        inputs=inputs,
        call_kwargs={"background_data": background},
        call_provenance={"background_data": {"source": "imagenet", "n_samples": 5}},
        input_spec=_image_input_spec((2, 3, 4, 4)),
        run_dir=tmp_path,
    )
    assert record is not None
    assert record.mode == "configured"
    assert record.source == "imagenet"
    assert record.n_samples == 5
    assert record.shape == (5, 3, 4, 4)


def test_bfloat16_baseline_renders_and_hashes(tmp_path: Path) -> None:
    record = build_baseline_record(
        explainer=_captum_ig(),
        inputs=torch.rand(2, 3, 4, 4, dtype=torch.bfloat16),
        call_kwargs={},
        call_provenance=None,
        input_spec=_image_input_spec((2, 3, 4, 4)),
        run_dir=tmp_path,
    )
    assert record is not None
    assert record.sha256  # bf16 hashed without crashing
    assert record.image_path is not None
    assert (tmp_path / record.image_path).exists()  # bf16 rendered without crashing


def test_render_cache_reuses_rendered_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    render_calls: list[Path] = []

    def _counting_render(baseline: torch.Tensor, run_dir: Path) -> Path:
        render_calls.append(run_dir)
        return _original_render_baseline_image(baseline, run_dir)

    monkeypatch.setattr("raitap.transparency.baselines._render_baseline_image", _counting_render)

    cache: dict[str, Path] = {}
    common = {
        "explainer": _captum_ig(),
        "inputs": torch.rand(1, 3, 4, 4),
        "call_kwargs": {},
        "call_provenance": None,
        "input_spec": _image_input_spec((1, 3, 4, 4)),
    }
    box0 = tmp_path / "box0"
    box1 = tmp_path / "box1"
    record0 = build_baseline_record(**common, run_dir=box0, render_cache=cache)
    record1 = build_baseline_record(**common, run_dir=box1, render_cache=cache)

    assert record0 is not None
    assert record1 is not None
    assert record0.sha256 == record1.sha256
    # matplotlib ran exactly once; the second box reused the rendered image.
    assert len(render_calls) == 1
    assert record0.image_path is not None
    assert record1.image_path is not None
    assert (box0 / record0.image_path).exists()
    assert (box1 / record1.image_path).exists()
    assert (box0 / "baseline.png").read_bytes() == (box1 / "baseline.png").read_bytes()


def test_montage_caption_only_when_capped() -> None:
    from raitap.transparency.baselines import _montage_caption

    assert _montage_caption(25, 200) == "Showing 25 of 200"
    assert _montage_caption(20, 20) is None  # all tiles shown -> no label
    assert _montage_caption(1, 1) is None


def test_multi_image_baseline_renders_grid_montage(tmp_path: Path) -> None:
    from PIL import Image

    background = torch.rand(6, 3, 8, 8)
    record = build_baseline_record(
        explainer=_shap_gradient(),
        inputs=torch.rand(1, 3, 8, 8),
        call_kwargs={"background_data": background},
        call_provenance={"background_data": {"source": "bg", "n_samples": 6}},
        input_spec=_image_input_spec((1, 3, 8, 8)),
        run_dir=tmp_path,
    )
    assert record is not None
    assert record.image_path is not None
    rendered = tmp_path / record.image_path
    assert rendered.exists()
    # 6 images -> 5 cols x 2 rows grid -> wider than a single square tile.
    with Image.open(rendered) as image:
        assert image.width > image.height
    # The descriptor still records the full set, not just the previewed tiles.
    assert record.shape == (6, 3, 8, 8)
    assert record.n_samples == 6


def test_shap_input_batch_default_when_absent(tmp_path: Path) -> None:
    inputs = torch.rand(3, 3, 4, 4)
    record = build_baseline_record(
        explainer=_shap_gradient(),
        inputs=inputs,
        call_kwargs={},
        call_provenance=None,
        input_spec=_image_input_spec((3, 3, 4, 4)),
        run_dir=tmp_path,
    )
    assert record is not None
    assert record.mode == "input_batch"
    assert record.shape == (3, 3, 4, 4)  # the full input batch


def test_no_record_for_algorithm_without_baseline(tmp_path: Path) -> None:
    saliency = SimpleNamespace(
        algorithm="Saliency",
        baseline_kwarg="baselines",
        baseline_defaults={"IntegratedGradients": "zero"},
    )
    record = build_baseline_record(
        explainer=saliency,
        inputs=torch.rand(2, 3, 4, 4),
        call_kwargs={},
        call_provenance=None,
        input_spec=_image_input_spec((2, 3, 4, 4)),
        run_dir=tmp_path,
    )
    assert record is None


def test_no_record_when_family_takes_no_baseline(tmp_path: Path) -> None:
    explainer = SimpleNamespace(algorithm="X", baseline_kwarg=None, baseline_defaults={})
    record = build_baseline_record(
        explainer=explainer,
        inputs=torch.rand(2, 3, 4, 4),
        call_kwargs={},
        call_provenance=None,
        input_spec=_image_input_spec((2, 3, 4, 4)),
        run_dir=tmp_path,
    )
    assert record is None


def test_no_image_for_non_image_modality(tmp_path: Path) -> None:
    record = build_baseline_record(
        explainer=_captum_ig(),
        inputs=torch.rand(2, 4),
        call_kwargs={},
        call_provenance=None,
        input_spec=_tabular_input_spec((2, 4)),
        run_dir=tmp_path,
    )
    assert record is not None
    assert record.mode == "zero"
    assert record.image_path is None
    assert not (tmp_path / "baseline.png").exists()
