"""Public namespaced decorator surface for registering RAITAP adapters.

This is the **public / plugin-author** surface. Third-party plugins decorate
with ``@adapters.<family>(...)``::

    from raitap import adapters

    @adapters.robustness(registry_name="myattack", algorithm_registry={...})
    class MyAttackAssessor(EmpiricalAttackAssessor): ...

In-tree adapters do **not** import this facade — they decorate with the bare
family decorator imported directly from their sibling ``registration.py`` (e.g.
``from .registration import robustness_adapter``) to avoid an import cycle
(this module imports every family package, which imports its leaf adapters).

Each attribute is the real, fully-typed family decorator (not a dynamic
attribute), so pyright checks kwargs at the decoration site. Importing this
module stays torch-free: the underlying ``registration.py`` modules defer torch
via ``raitap.utils.lazy.lazy_import``.
"""

from __future__ import annotations

from raitap.metrics.registration import metrics_adapter as metrics
from raitap.reporting.registration import reporter
from raitap.robustness.assessors.registration import robustness_adapter as robustness
from raitap.tracking.registration import tracker
from raitap.transparency.explainers.registration import transparency_adapter as transparency
from raitap.transparency.visualisers.image_rendering import image_renderer

__all__ = ["image_renderer", "metrics", "reporter", "robustness", "tracker", "transparency"]
