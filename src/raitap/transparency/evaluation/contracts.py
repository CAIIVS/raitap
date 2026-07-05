"""Typed contracts for Quantus-based explanation-quality grading (#341)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from raitap.tracking.base_tracker import Trackable

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from raitap.tracking.base_tracker import BaseTracker


class QuantusCategory(StrEnum):
    FAITHFULNESS = "faithfulness"
    ROBUSTNESS = "robustness"
    COMPLEXITY = "complexity"
    LOCALISATION = "localisation"
    RANDOMISATION = "randomisation"
    AXIOMATIC = "axiomatic"


class EvalRequirement(StrEnum):
    ATTRIBUTIONS = "attributions"
    MODEL = "model"
    RE_EXPLAIN = "re_explain"
    SEGMENTATION = "segmentation"
    BASELINE = "baseline"


class EvaluationIncompatible(ValueError):  # noqa: N818
    """A metric's ``requires`` are not satisfiable for an explanation."""


EvalInvoker = Callable[..., list[float]]


@dataclass(frozen=True)
class QuantusMetricSpec:
    category: QuantusCategory
    quantus_cls: str
    requires: frozenset[EvalRequirement]
    higher_is_better: bool | None = None
    default_kwargs: Mapping[str, Any] = field(default_factory=dict)
    invoker: EvalInvoker | None = None


@dataclass
class EvaluationScore:
    metric: str
    category: QuantusCategory
    values: list[float]
    aggregate: float | None
    higher_is_better: bool | None


@dataclass
class SkippedMetric:
    metric: str
    missing: frozenset[EvalRequirement]
    message: str


@dataclass
class EvaluationResult(Trackable):
    explanation_name: str | None
    adapter_target: str
    algorithm: str
    scores: list[EvaluationScore] = field(default_factory=list)
    skipped: list[SkippedMetric] = field(default_factory=list)
    run_dir: Path | None = None

    def log(self, tracker: BaseTracker | None = None, **kwargs: Any) -> None:
        if tracker is None:
            return
        metrics = {s.metric: s.aggregate for s in self.scores if s.aggregate is not None}
        if metrics:
            tracker.log_metrics(metrics, prefix="explanation_quality")
