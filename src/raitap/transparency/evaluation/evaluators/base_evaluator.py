"""Base class for explanation-quality evaluators (#341)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from raitap._adapters import AdapterMixin

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from raitap.transparency.evaluation.contracts import EvaluationResult, QuantusMetricSpec

    # Forward reference: Wave-3 work (not yet created on this branch). Under
    # ``from __future__ import annotations`` this doesn't break at runtime;
    # pyright still resolves both branches of ``TYPE_CHECKING`` statically.
    from raitap.transparency.evaluation.semantics import (  # pyright: ignore[reportMissingImports]
        EvaluationContext,
    )


class BaseEvaluator(AdapterMixin, ABC):
    algorithm_registry: ClassVar[Mapping[str, QuantusMetricSpec]]

    @abstractmethod
    def evaluate(self, ctx: EvaluationContext, *, run_dir: Path | None) -> EvaluationResult:
        """Grade one explanation, returning per-metric scores + skipped records."""
