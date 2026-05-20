"""Public namespaced decorator surface for registering visualisers.

from raitap import visualisers

@visualisers.transparency(registry_name="my_viz", supported_scopes=...)
class MyVisualiser(BaseVisualiser): ...
"""

from __future__ import annotations

from raitap.robustness.visualisers.registration import robustness_visualiser as robustness
from raitap.transparency.visualisers.registration import transparency_visualiser as transparency

__all__ = ["robustness", "transparency"]
