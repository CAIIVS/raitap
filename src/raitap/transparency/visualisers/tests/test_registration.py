"""Decorator integration test: a decorated stub visualiser lands in the
``_unscoped`` builder pool, since visualisers have no Hydra family group."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from matplotlib.figure import Figure

from raitap.transparency.visualisers.base_visualiser import BaseVisualiser
from raitap.transparency.visualisers.registration import register_transparency_visualiser

if TYPE_CHECKING:
    import torch

    from raitap.transparency.contracts import VisualisationContext


# Defined at module scope so ``hydra_zen.builds(...)`` can resolve the class's
# ``__module__`` — nested-in-function classes would not be importable and the
# ``_register_core`` swallow would silently skip registration.
@register_transparency_visualiser(
    registry_name="_stub_viz",
)
class _StubViz(BaseVisualiser):
    def __init__(self, max_samples: int = 4) -> None:
        self.max_samples = max_samples

    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        *,
        context: VisualisationContext | None = None,
        **kwargs: Any,
    ) -> Figure:
        del attributions, inputs, context, kwargs
        return Figure()


def test_register_transparency_visualiser_lands_in_unscoped_pool() -> None:
    from raitap._adapters import _BUILDERS

    assert "_stub_viz" in _BUILDERS["_unscoped"]
