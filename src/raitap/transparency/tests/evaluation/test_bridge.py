from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from raitap.transparency.contracts import (
    ExplanationOutputSpace,
    ExplanationPayloadKind,
    ExplanationScope,
    ExplanationSemantics,
    OutputSpaceSpec,
    ScopeDefinitionStep,
    TensorLayout,
)
from raitap.transparency.evaluation.bridge import (
    derive_channel_first,
    resolve_target,
    to_quantus_arrays,
)
from raitap.transparency.results import ExplanationResult

if TYPE_CHECKING:
    from pathlib import Path


def _make_result(
    tmp_path: Path,
    *,
    target: int | list[int] | None,
    space: ExplanationOutputSpace,
    layout: TensorLayout | None = None,
) -> ExplanationResult:
    return ExplanationResult(
        attributions=torch.randn(4, 3, 8, 8),
        inputs=torch.randn(4, 3, 8, 8),
        run_dir=tmp_path,
        experiment_name=None,
        adapter_target="raitap.transparency.CaptumExplainer",
        algorithm="IntegratedGradients",
        call_kwargs={"target": target},
        semantics=ExplanationSemantics(
            scope=ExplanationScope.LOCAL,
            scope_definition_step=ScopeDefinitionStep.EXPLAINER_OUTPUT,
            payload_kind=ExplanationPayloadKind.ATTRIBUTIONS,
            method_families=frozenset(),
            target=None,
            sample_selection=None,
            input_spec=None,
            output_space=OutputSpaceSpec(space=space, shape=None, layout=layout),
        ),
    )


def test_channel_first_true_for_image_space(tmp_path: Path) -> None:
    r = _make_result(tmp_path, target=3, space=ExplanationOutputSpace.IMAGE_SPATIAL_MAP)
    assert derive_channel_first(r) is True


def test_channel_first_false_for_non_image_space(tmp_path: Path) -> None:
    r = _make_result(tmp_path, target=3, space=ExplanationOutputSpace.INPUT_FEATURES)
    assert derive_channel_first(r) is False


def test_channel_first_true_for_nchw_input_features(tmp_path: Path) -> None:
    """Plain gradient attributions (Saliency/IG) on images stay INPUT_FEATURES
    space but keep the NCHW layout — Quantus still needs channel_first=True (#341)."""
    r = _make_result(
        tmp_path,
        target=3,
        space=ExplanationOutputSpace.INPUT_FEATURES,
        layout=TensorLayout.BATCH_CHANNEL_HEIGHT_WIDTH,
    )
    assert derive_channel_first(r) is True


def test_resolve_scalar_target_broadcasts(tmp_path: Path) -> None:
    r = _make_result(tmp_path, target=3, space=ExplanationOutputSpace.IMAGE_SPATIAL_MAP)
    assert resolve_target(r) == 3
    arrays = to_quantus_arrays(r, target=3)
    assert arrays.x_batch.shape == (4, 3, 8, 8)
    assert arrays.y_batch.tolist() == [3, 3, 3, 3]
    assert arrays.a_batch.dtype == np.float32


def test_resolve_target_none_when_absent(tmp_path: Path) -> None:
    r = _make_result(tmp_path, target=None, space=ExplanationOutputSpace.IMAGE_SPATIAL_MAP)
    assert resolve_target(r) is None


def test_to_quantus_arrays_list_target(tmp_path: Path) -> None:
    r = _make_result(tmp_path, target=[0, 1, 2, 3], space=ExplanationOutputSpace.IMAGE_SPATIAL_MAP)
    arrays = to_quantus_arrays(r, target=[0, 1, 2, 3])
    assert arrays.y_batch.tolist() == [0, 1, 2, 3]
    assert arrays.x_batch.dtype == np.float32


def test_explainer_to_explain_func_roundtrips_numpy() -> None:
    from raitap.transparency.evaluation.bridge import explainer_to_explain_func

    class _Stub:
        def compute_attributions(
            self, model: object, inputs: torch.Tensor, target: object = None
        ) -> torch.Tensor:
            del model, target
            return inputs * 2.0

    fn = explainer_to_explain_func(_Stub(), torch.device("cpu"))  # type: ignore[arg-type]
    x = np.ones((2, 3), dtype=np.float32)
    out = fn(model=object(), inputs=x, targets=np.array([0, 1]))
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (2, 3)
    assert np.allclose(out, 2.0)
