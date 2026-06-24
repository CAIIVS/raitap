"""Label parser family package.

Importing this package fires the ``@label_parser`` decorator on every
in-tree parser module, registering them with the hydra-zen store. Each
concrete parser is re-exported here so the short ``_target_`` form (a bare
class name resolved against ``raitap.data.label_parsers.``) instantiates,
mirroring how ``raitap.metrics`` re-exports its metric computers.
"""

from __future__ import annotations

from .directory import DirectoryLabelParser
from .tabular import TabularLabelParser  # pyright: ignore[reportUnusedImport]

__all__ = ["DirectoryLabelParser", "TabularLabelParser"]
