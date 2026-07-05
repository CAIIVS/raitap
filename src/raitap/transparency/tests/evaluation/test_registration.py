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


_REGISTRY = {
    "sparseness": QuantusMetricSpec(
        QuantusCategory.COMPLEXITY,
        "Sparseness",
        frozenset({EvalRequirement.ATTRIBUTIONS}),
        higher_is_better=True,
    )
}


# Defined at module scope so ``hydra_zen.builds(...)`` can resolve the class's
# ``__module__`` and it actually registers into ``_BUILDERS["_unscoped"]`` —
# nested-in-function classes are not importable and ``_register_core`` swallows
# their registration silently.
#
# Deliberately NO ``extra`` kwarg: a test-only adapter that carried one would
# land in the global ``ADAPTER_EXTRAS`` under a class name the static AST
# scanner cannot see (it only scans ``src/``), breaking
# ``deps/tests/test_static_scan.py::test_runtime_extras_subset_of_static_scan``.
# The sibling visualiser stub (``visualisers/tests/test_registration.py``)
# follows the same extra-less convention. The real evaluator's ``extra="quantus"``
# is exercised end-to-end by the deps inference tests.
@transparency_evaluator(
    registry_name="dummy_eval",
    library="quantus",
    algorithm_registry=_REGISTRY,
)
class DummyEval(BaseEvaluator):
    def evaluate(self, ctx: Any, *, run_dir: Path | None) -> EvaluationResult:
        raise NotImplementedError


def test_decorator_sets_registry_and_identity() -> None:
    assert DummyEval.registry_name == "dummy_eval"
    assert DummyEval.library == "quantus"
    assert "sparseness" in DummyEval.algorithm_registry


def test_transparency_evaluator_lands_in_unscoped_pool() -> None:
    from raitap._adapters import _BUILDERS

    assert "dummy_eval" in _BUILDERS["_unscoped"]
