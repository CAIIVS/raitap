# src/raitap/transparency/explainers/tests/test_shap_parity.py
"""Tier-1 parity: raitap's ShapExplainer returns the same numbers as a direct
shap call for the same model/inputs/config/seed. Catches dropped kwargs, wrong
target wiring, or stray mutation in the data path. SHAP needs this most: it does
real post-processing (multi-class class-stacking + per-sample target selection)
between the library call and the persisted artefact, unlike the near
pass-through Captum wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch

from raitap.testing import make_tiny_classifier

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [pytest.mark.e2e, pytest.mark.parity]

_RTOL = 1e-5
_ATOL = 1e-6


@pytest.fixture
def _shap() -> Any:
    return pytest.importorskip("shap")


def _stack_select(shap_values: Any, *, target: int | list[int]) -> torch.Tensor:
    """Replicate the wrapper's documented post-processing explicitly.

    1. Normalise to a class-last tensor: a list of per-class arrays is stacked on
       ``dim=-1``; an ndarray/tensor already carries the class axis last.
    2. Select the target: an int indexes the class axis directly; a per-sample
       list advanced-indexes one class per sample.

    Mirrors ``shap_explainer._select_target_attributions`` but spelled out here
    so the reference path stays independent of the code under test.
    """
    if isinstance(shap_values, list):
        stacked = torch.stack(
            [v if isinstance(v, torch.Tensor) else torch.from_numpy(v) for v in shap_values],
            dim=-1,
        )
    elif isinstance(shap_values, torch.Tensor):
        stacked = shap_values
    else:
        stacked = torch.from_numpy(shap_values)

    if isinstance(target, int):
        return stacked[..., target]

    target_tensor = torch.tensor(target, dtype=torch.long)
    batch_indices = torch.arange(stacked.shape[0], dtype=torch.long)
    return stacked[batch_indices, ..., target_tensor]


def test_gradient_explainer_int_target_matches_direct_call(
    _shap: Any, seeded: Callable[..., None]
) -> None:
    """int target → class-stack + int-select must match a direct shap call."""
    from raitap.transparency.explainers import ShapExplainer

    seeded()
    model = make_tiny_classifier(seed=0)
    x = torch.randn(2, 3, 8, 8)
    bg = torch.randn(2, 3, 8, 8)

    seeded()  # GradientExplainer path-samples via numpy RNG; pin before each path.
    out = ShapExplainer("GradientExplainer").compute_attributions(
        model, x, background_data=bg, target=0
    )

    seeded()
    sv = _shap.GradientExplainer(model, bg).shap_values(x)
    ref = _stack_select(sv, target=0)

    assert torch.allclose(out.detach().cpu(), ref.detach().cpu(), rtol=_RTOL, atol=_ATOL)


def test_gradient_explainer_per_sample_target_matches_direct_call(
    _shap: Any, seeded: Callable[..., None]
) -> None:
    """Per-sample list target → class-stack + advanced-index select must match a
    direct shap call. Exercises the ``_select_target_attributions`` per-sample
    branch — the wrapper logic with the most surface for wiring bugs."""
    from raitap.transparency.explainers import ShapExplainer

    seeded()
    model = make_tiny_classifier(seed=0)
    x = torch.randn(2, 3, 8, 8)
    bg = torch.randn(2, 3, 8, 8)

    seeded()
    out = ShapExplainer("GradientExplainer").compute_attributions(
        model, x, background_data=bg, target=[0, 1]
    )

    seeded()
    sv = _shap.GradientExplainer(model, bg).shap_values(x)
    ref = _stack_select(sv, target=[0, 1])

    assert torch.allclose(out.detach().cpu(), ref.detach().cpu(), rtol=_RTOL, atol=_ATOL)
