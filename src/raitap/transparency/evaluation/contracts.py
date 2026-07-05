"""Typed contracts for Quantus-based explanation-quality grading (#341)."""

from __future__ import annotations

import json
from collections.abc import (
    Callable,
    Mapping,
)
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path  # noqa: TC003
from typing import Any

from raitap.tracking.base_tracker import BaseTracker, Trackable


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

    def write_artifacts(self) -> None:
        """Write ``<run_dir>/evaluations.json``; a no-op when ``run_dir`` is unset."""
        if self.run_dir is None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "explanation_name": self.explanation_name,
            "algorithm": self.algorithm,
            "adapter_target": self.adapter_target,
            "scores": [
                {
                    "metric": score.metric,
                    "category": score.category.value,
                    "aggregate": score.aggregate,
                    "values": score.values,
                    "higher_is_better": score.higher_is_better,
                }
                for score in self.scores
            ],
            "skipped": [
                {
                    "metric": skip.metric,
                    "missing": sorted(req.value for req in skip.missing),
                    "message": skip.message,
                }
                for skip in self.skipped
            ],
        }
        with (self.run_dir / "evaluations.json").open("w") as handle:
            json.dump(payload, handle, indent=2)
