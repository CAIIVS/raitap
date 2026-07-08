"""Label parser family package.

Importing this package fires the ``@label_parser`` decorator on every
in-tree parser module, registering them with the hydra-zen store. Each
concrete parser is re-exported here for direct Python-side access (e.g.
``from raitap.data.label_parsers import CocoLabelParser``), mirroring how
``raitap.metrics`` re-exports its metric computers. YAML selection goes
through ``use: <registry_name>`` (e.g. ``use: coco``), resolved against the
trusted registry (:mod:`raitap.configs.registry_resolve`) independently of
this package's namespace.
"""

from __future__ import annotations

from .coco import CocoLabelParser  # pyright: ignore[reportUnusedImport]
from .detection_json import DetectionJsonLabelParser  # pyright: ignore[reportUnusedImport]
from .directory import DirectoryLabelParser  # pyright: ignore[reportUnusedImport]
from .tabular import TabularLabelParser  # pyright: ignore[reportUnusedImport]
from .voc import VocLabelParser  # pyright: ignore[reportUnusedImport]
from .yolo import YoloLabelParser  # pyright: ignore[reportUnusedImport]

__all__ = [
    "CocoLabelParser",
    "DetectionJsonLabelParser",
    "DirectoryLabelParser",
    "TabularLabelParser",
    "VocLabelParser",
    "YoloLabelParser",
]
