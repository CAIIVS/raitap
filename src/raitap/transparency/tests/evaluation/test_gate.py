from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("torch")

import torch

from raitap.transparency.evaluation.contracts import (
    EvalRequirement,
    QuantusCategory,
    QuantusMetricSpec,
    SkippedMetric,
)
from raitap.transparency.evaluation.semantics import (
    EvaluationContext,
    ResolvedMetric,
    resolve_metric,
)

if TYPE_CHECKING:
    from pathlib import Path


def _ctx(
    tmp_path: Path, *, explainer: Any | None = None, model: Any | None = None
) -> EvaluationContext:
    from raitap.transparency.contracts import ExplanationOutputSpace
    from raitap.transparency.tests.evaluation.test_bridge import _make_result

    result = _make_result(tmp_path, target=1, space=ExplanationOutputSpace.IMAGE_SPATIAL_MAP)
    return EvaluationContext(
        result=result,
        model=model,
        device=torch.device("cpu"),
        explainer=explainer,
        masks=None,
        baseline=None,
        softmax=False,
    )


def test_complexity_always_resolves(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, model=object())
    spec = QuantusMetricSpec(
        QuantusCategory.COMPLEXITY,
        "Sparseness",
        frozenset({EvalRequirement.ATTRIBUTIONS}),
        higher_is_better=True,
    )
    out = resolve_metric("sparseness", spec, ctx)
    assert isinstance(out, ResolvedMetric)
    assert "a_batch" in out.call_kwargs


def test_robustness_skips_without_explainer(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, model=object(), explainer=None)
    spec = QuantusMetricSpec(
        QuantusCategory.ROBUSTNESS,
        "MaxSensitivity",
        frozenset(
            {EvalRequirement.ATTRIBUTIONS, EvalRequirement.MODEL, EvalRequirement.RE_EXPLAIN}
        ),
        higher_is_better=False,
    )
    out = resolve_metric("max_sensitivity", spec, ctx)
    assert isinstance(out, SkippedMetric)
    assert EvalRequirement.RE_EXPLAIN in out.missing
