"""Hydra SearchPathPlugin discovery shim — re-exports the real plugin.

Hydra scans the ``hydra_plugins`` namespace package for plugin classes at
import time. The actual implementation lives at
:class:`raitap.configs.searchpath.RaitapSearchPathPlugin`; this module just
makes it discoverable.
"""

from __future__ import annotations

from raitap.configs.searchpath import RaitapSearchPathPlugin

__all__ = ["RaitapSearchPathPlugin"]
