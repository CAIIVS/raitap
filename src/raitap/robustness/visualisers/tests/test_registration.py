"""Decorator integration test: a decorated stub robustness visualiser lands
in the ``_unscoped`` builder pool, since visualisers have no Hydra family
group."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap import visualisers
from raitap.robustness.visualisers.base_visualiser import BaseRobustnessVisualiser

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from raitap.robustness.contracts import RobustnessVisualisationContext
    from raitap.robustness.results import RobustnessResult


# Defined at module scope so hydra-zen's ``populate_full_signature`` builder can
# introspect a real import path. Function-local classes have a ``<locals>``
# ``__qualname__`` that ``builds(...)`` rejects with ``TypeError``, which
# ``_register_core`` then silently swallows.
@visualisers.robustness(
    registry_name="_stub_rob_viz",
)
class _StubRobViz(BaseRobustnessVisualiser):
    def __init__(self, max_samples: int = 4) -> None:
        self.max_samples = max_samples

    def visualise(
        self,
        result: RobustnessResult,
        *,
        context: RobustnessVisualisationContext,
        **kwargs: Any,
    ) -> Figure:
        del result, context, kwargs
        from matplotlib.figure import Figure as _Figure

        return _Figure()


def test_robustness_visualiser_lands_in_unscoped_pool() -> None:
    from raitap._adapters import _BUILDERS

    assert "_stub_rob_viz" in _BUILDERS["_unscoped"]


def test_robustness_capability_fields_via_decorator() -> None:
    from raitap.robustness.contracts import MethodKind
    from raitap.robustness.visualisers.base_visualiser import BaseRobustnessVisualiser
    from raitap.robustness.visualisers.registration import robustness_visualiser

    @robustness_visualiser(
        registry_name="_stub_rob_caps",
        supported_method_kinds=frozenset({MethodKind.EMPIRICAL_ATTACK}),
        embeds_clean_input=True,
    )
    class _StubRob(BaseRobustnessVisualiser):
        def visualise(self, *a, **k):  # type: ignore[no-untyped-def]
            ...

    assert _StubRob.supported_method_kinds == frozenset({MethodKind.EMPIRICAL_ATTACK})
    assert _StubRob.embeds_clean_input is True
    assert _StubRob.embeds_perturbation_map == BaseRobustnessVisualiser.embeds_perturbation_map
