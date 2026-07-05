"""Base class for explanation-quality evaluators (#341)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from raitap._adapters import AdapterMixin

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from raitap.transparency.evaluation.contracts import EvaluationResult, QuantusMetricSpec
    from raitap.transparency.evaluation.semantics import EvaluationContext


class BaseEvaluator(AdapterMixin, ABC):
    algorithm_registry: ClassVar[Mapping[str, QuantusMetricSpec]]

    @abstractmethod
    def evaluate(self, ctx: EvaluationContext, *, run_dir: Path | None) -> EvaluationResult:
        """Grade one explanation, returning per-metric scores + skipped records."""
