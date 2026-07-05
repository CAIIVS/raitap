"""Requirement resolution + compat gate for quantus grading (#341)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from raitap.transparency.evaluation.bridge import (
    derive_channel_first,
    explainer_to_explain_func,
    resolve_target,
    to_quantus_arrays,
)
from raitap.transparency.evaluation.contracts import EvalRequirement, SkippedMetric
from raitap.transparency.explainers.base_explainer import AttributionOnlyExplainer

if TYPE_CHECKING:
    import torch

    from raitap.transparency.evaluation.contracts import QuantusMetricSpec
    from raitap.transparency.explainers.base_explainer import BaseExplainer
    from raitap.transparency.results import ExplanationResult


@dataclass
class ResolvedMetric:
    spec: QuantusMetricSpec
    call_kwargs: dict[str, Any]


@dataclass
class EvalInvokeCtx:
    evaluator: Any
    quantus: Any
    metric_cls: Any
    call_kwargs: dict[str, Any]


@dataclass
class EvaluationContext:
    result: ExplanationResult
    model: Any
    device: torch.device
    explainer: BaseExplainer | None
    masks: Any | None
    baseline: Any | None
    softmax: bool

    def available_requirements(self) -> frozenset[EvalRequirement]:
        reqs = {EvalRequirement.ATTRIBUTIONS}
        if self.model is not None:
            reqs.add(EvalRequirement.MODEL)
        if isinstance(self.explainer, AttributionOnlyExplainer):
            reqs.add(EvalRequirement.RE_EXPLAIN)
        if self.masks is not None:
            reqs.add(EvalRequirement.SEGMENTATION)
        if self.baseline is not None:
            reqs.add(EvalRequirement.BASELINE)
        return frozenset(reqs)

    def gather(self, spec: QuantusMetricSpec) -> dict[str, Any]:
        arrays = to_quantus_arrays(self.result, target=resolve_target(self.result))
        kw: dict[str, Any] = {
            "model": self.model,
            "x_batch": arrays.x_batch,
            "y_batch": arrays.y_batch,
            "a_batch": arrays.a_batch,
            "device": str(self.device),
            "channel_first": derive_channel_first(self.result),
            "softmax": self.softmax,
        }
        if EvalRequirement.RE_EXPLAIN in spec.requires and isinstance(
            self.explainer, AttributionOnlyExplainer
        ):
            kw["explain_func"] = explainer_to_explain_func(self.explainer, self.device)
        if EvalRequirement.SEGMENTATION in spec.requires:
            kw["s_batch"] = self.masks
        return kw


def resolve_metric(
    metric: str, spec: QuantusMetricSpec, ctx: EvaluationContext
) -> ResolvedMetric | SkippedMetric:
    missing = spec.requires - ctx.available_requirements()
    if missing:
        needed = sorted(m.value for m in missing)
        return SkippedMetric(
            metric,
            frozenset(missing),
            f"quantus metric {metric!r} needs {needed}, not available for this explanation",
        )
    return ResolvedMetric(spec, ctx.gather(spec))
