"""SHAP modern-path normalization helpers (#267)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from raitap.transparency.contracts import InputKind, InputSpec, TensorLayout
from raitap.transparency.explainers.shap_explainer import (
    _build_masker,
    _modern_predict_fn,
    _normalise_modern_explanation,
)


def _tab_spec() -> InputSpec:
    return InputSpec(
        kind=InputKind.TABULAR,
        shape=(2, 8),
        layout=TensorLayout.BATCH_FEATURE,
        feature_names=None,
        metadata=None,
    )


def _img_spec() -> InputSpec:
    return InputSpec(
        kind=InputKind.IMAGE,
        shape=(2, 3, 8, 8),
        layout=TensorLayout.BATCH_CHANNEL_HEIGHT_WIDTH,
        feature_names=None,
        metadata=None,
    )


def test_normalise_tabular_selects_class_and_casts() -> None:
    values = np.random.randn(2, 8, 5).astype("float64")  # (B, F, K)
    out = _normalise_modern_explanation(values, input_spec=_tab_spec(), target=1)
    assert out.shape == (2, 8)
    assert out.dtype == torch.float32


def test_normalise_image_permutes_nhwc_to_nchw() -> None:
    values = np.random.randn(2, 8, 8, 3, 5).astype("float64")  # (B, H, W, C, K)
    out = _normalise_modern_explanation(values, input_spec=_img_spec(), target=0)
    assert out.shape == (2, 3, 8, 8)  # NCHW
    assert out.dtype == torch.float32


def test_normalise_tabular_no_target_keeps_class_axis() -> None:
    # With no target, the class axis is kept (matches _select_target_attributions semantics).
    values = np.random.randn(2, 8, 5).astype("float64")
    out = _normalise_modern_explanation(values, input_spec=_tab_spec(), target=None)
    assert out.shape == (2, 8, 5)
    assert out.dtype == torch.float32


def test_modern_predict_fn_image_permutes_nhwc_to_nchw() -> None:
    # The masker hands NHWC; the wrapper must feed the model NCHW.
    seen: list[tuple[int, ...]] = []

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        seen.append(tuple(x.shape))
        return torch.zeros(x.shape[0], 5)

    predict = _modern_predict_fn(model_fn, input_spec=_img_spec())
    out = predict(np.zeros((2, 8, 8, 3), dtype="float32"))  # NHWC
    assert seen == [(2, 3, 8, 8)]  # model saw NCHW
    assert out.shape == (2, 5)


def test_modern_predict_fn_tabular_passthrough() -> None:
    seen: list[tuple[int, ...]] = []

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        seen.append(tuple(x.shape))
        return torch.zeros(x.shape[0], 5)

    predict = _modern_predict_fn(model_fn, input_spec=_tab_spec())
    predict(np.zeros((2, 8), dtype="float32"))
    assert seen == [(2, 8)]  # no permute


def test_build_masker_rejects_unsupported_modality() -> None:
    spec = InputSpec(
        kind=InputKind.TEXT, shape=(2, 16), layout=None, feature_names=None, metadata=None
    )
    # Raises before touching ``shap``, so a ``None`` shap arg is fine.
    with pytest.raises(ValueError, match="support image and tabular"):
        _build_masker(None, input_spec=spec, background=None)
