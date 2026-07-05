from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap.transparency.evaluation.contracts import (
    EvalRequirement,
    EvaluationResult,
    QuantusCategory,
    QuantusMetricSpec,
)
from raitap.transparency.evaluation.evaluators.base_evaluator import BaseEvaluator
from raitap.transparency.evaluation.evaluators.registration import transparency_evaluator

if TYPE_CHECKING:
    from pathlib import Path


# Defined at module scope so ``hydra_zen.builds(...)`` can resolve the class's
# ``__module__`` — nested-in-function classes would not be importable and the
# ``_register_core`` swallow would silently skip registration.
@transparency_evaluator(
    registry_name="dummy_eval",
    library="quantus",
    extra="quantus",
    algorithm_registry={
        "sparseness": QuantusMetricSpec(
            QuantusCategory.COMPLEXITY,
            "Sparseness",
            frozenset({EvalRequirement.ATTRIBUTIONS}),
            higher_is_better=True,
        )
    },
)
class DummyEval(BaseEvaluator):
    def evaluate(self, ctx: Any, *, run_dir: Path | None) -> EvaluationResult:
        raise NotImplementedError


def test_decorator_sets_registry_and_identity() -> None:
    assert DummyEval.registry_name == "dummy_eval"
    assert DummyEval.library == "quantus"
    assert DummyEval.extra == "quantus"
    assert "sparseness" in DummyEval.algorithm_registry


def test_transparency_evaluator_lands_in_unscoped_pool() -> None:
    from raitap._adapters import _BUILDERS

    assert "dummy_eval" in _BUILDERS["_unscoped"]
