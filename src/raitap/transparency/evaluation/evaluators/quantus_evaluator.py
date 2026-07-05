"""Quantus explanation-quality evaluator: all 6 categories (#341)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap.transparency.evaluation.contracts import (
    EvalRequirement as Req,
)
from raitap.transparency.evaluation.contracts import (
    EvaluationResult,
    EvaluationScore,
)
from raitap.transparency.evaluation.contracts import (
    QuantusCategory as C,
)
from raitap.transparency.evaluation.contracts import (
    QuantusMetricSpec as Spec,
)
from raitap.transparency.evaluation.evaluators.base_evaluator import BaseEvaluator
from raitap.transparency.evaluation.evaluators.registration import transparency_evaluator
from raitap.transparency.evaluation.semantics import ResolvedMetric, resolve_metric

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

    from raitap.transparency.evaluation.contracts import SkippedMetric
    from raitap.transparency.evaluation.semantics import EvaluationContext

_REGISTRY: dict[str, Spec] = {
    # Faithfulness -- a_batch + model re-inference (no re-explain)
    "faithfulness_correlation": Spec(
        C.FAITHFULNESS,
        "FaithfulnessCorrelation",
        frozenset({Req.ATTRIBUTIONS, Req.MODEL}),
        higher_is_better=True,
    ),
    "pixel_flipping": Spec(
        C.FAITHFULNESS,
        "PixelFlipping",
        frozenset({Req.ATTRIBUTIONS, Req.MODEL}),
        higher_is_better=False,
    ),
    "road": Spec(
        C.FAITHFULNESS,
        "ROAD",
        frozenset({Req.ATTRIBUTIONS, Req.MODEL}),
        higher_is_better=False,
    ),
    # Complexity -- pure a_batch
    "sparseness": Spec(
        C.COMPLEXITY,
        "Sparseness",
        frozenset({Req.ATTRIBUTIONS}),
        higher_is_better=True,
    ),
    "complexity": Spec(
        C.COMPLEXITY,
        "Complexity",
        frozenset({Req.ATTRIBUTIONS}),
        higher_is_better=False,
    ),
    # Robustness -- needs re-explain
    "max_sensitivity": Spec(
        C.ROBUSTNESS,
        "MaxSensitivity",
        frozenset({Req.ATTRIBUTIONS, Req.MODEL, Req.RE_EXPLAIN}),
        higher_is_better=False,
    ),
    "avg_sensitivity": Spec(
        C.ROBUSTNESS,
        "AvgSensitivity",
        frozenset({Req.ATTRIBUTIONS, Req.MODEL, Req.RE_EXPLAIN}),
        higher_is_better=False,
    ),
    # Localisation -- needs segmentation masks
    "pointing_game": Spec(
        C.LOCALISATION,
        "PointingGame",
        frozenset({Req.ATTRIBUTIONS, Req.SEGMENTATION}),
        higher_is_better=True,
    ),
    "relevance_rank_accuracy": Spec(
        C.LOCALISATION,
        "RelevanceRankAccuracy",
        frozenset({Req.ATTRIBUTIONS, Req.SEGMENTATION}),
        higher_is_better=True,
    ),
    # Randomisation -- needs re-explain
    "model_parameter_randomisation": Spec(
        C.RANDOMISATION,
        "ModelParameterRandomisation",
        frozenset({Req.ATTRIBUTIONS, Req.MODEL, Req.RE_EXPLAIN}),
        higher_is_better=None,
    ),
    "random_logit": Spec(
        C.RANDOMISATION,
        "RandomLogit",
        frozenset({Req.ATTRIBUTIONS, Req.MODEL, Req.RE_EXPLAIN}),
        higher_is_better=None,
    ),
    # Axiomatic
    "completeness": Spec(
        C.AXIOMATIC,
        "Completeness",
        frozenset({Req.ATTRIBUTIONS, Req.MODEL}),
        higher_is_better=None,
    ),
    "input_invariance": Spec(
        C.AXIOMATIC,
        "InputInvariance",
        frozenset({Req.ATTRIBUTIONS, Req.MODEL, Req.RE_EXPLAIN}),
        higher_is_better=None,
    ),
}


def _aggregate(values: list[float]) -> float | None:
    finite = [v for v in values if v == v]  # drop NaN
    return sum(finite) / len(finite) if finite else None


@transparency_evaluator(
    registry_name="quantus",
    library="quantus",
    extra="quantus",
    algorithm_registry=_REGISTRY,
)
class QuantusEvaluator(BaseEvaluator):
    def __init__(
        self,
        metrics: list[str] | None = None,
        *,
        constructor: dict[str, dict[str, Any]] | None = None,
        softmax: bool = False,
        **kwargs: Any,
    ) -> None:
        # ``call`` and ``raitap`` are accepted via **kwargs rather than named
        # keyword-only params: naming them explicitly would collide with the
        # ``zen_meta={"call": {}, "raitap": {}}`` hydra-zen adds for every
        # family-less (``family=None``) adapter builder in ``_adapters.py``
        # (hydra-zen forbids zen_meta names that also exist in the target's
        # signature). Extracting them here still lets ``EvaluationConfig``
        # (which always carries ``call``/``raitap`` fields, defaulting to
        # ``{}``) be instantiated directly via ``hydra.utils.instantiate``.
        call = kwargs.pop("call", None)
        raitap = kwargs.pop("raitap", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"QuantusEvaluator got unexpected keyword argument(s): {unexpected}")
        self.metrics = metrics
        self.constructor = constructor or {}
        self.call = call or {}
        raitap_opts = raitap or {}
        self.softmax = bool(raitap_opts.get("softmax", softmax))

    def evaluate(self, ctx: EvaluationContext, *, run_dir: Path | None) -> EvaluationResult:
        quantus = self._lazy_import()
        selected = self.metrics or list(self.algorithm_registry)
        scores: list[EvaluationScore] = []
        skipped: list[SkippedMetric] = []
        for key in selected:
            spec = self.algorithm_registry[key]
            resolved = resolve_metric(key, spec, ctx)
            if not isinstance(resolved, ResolvedMetric):
                skipped.append(resolved)
                continue
            values = self._run_metric(quantus, key, spec, resolved)
            scores.append(
                EvaluationScore(
                    key, spec.category, values, _aggregate(values), spec.higher_is_better
                )
            )
        return EvaluationResult(
            explanation_name=ctx.result.name,
            adapter_target=ctx.result.adapter_target,
            algorithm=ctx.result.algorithm,
            scores=scores,
            skipped=skipped,
            run_dir=run_dir,
        )

    def _run_metric(
        self, quantus: ModuleType, key: str, spec: Spec, resolved: ResolvedMetric
    ) -> list[float]:
        metric_cls = getattr(quantus, spec.quantus_cls)
        init_kwargs = {
            **spec.default_kwargs,
            **self.constructor.get(key, {}),
            "disable_warnings": True,
        }
        metric = metric_cls(**init_kwargs)
        with self._rethrow():
            raw = metric(**{**resolved.call_kwargs, **self.call})
        return [float(v) for v in raw]
