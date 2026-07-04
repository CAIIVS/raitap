"""Family decorator for label-parser adapters.

Mirrors ``raitap.metrics.registration`` exactly, with group ``data/labels``
and ``package_style="flat"`` so composed configs land at ``cfg.data.labels``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Unpack

from raitap._adapters import AdapterDecoratorOptions, FamilyConfig, _register_core
from raitap.configs.schema import LabelsConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from raitap.data.label_parsers.base import LabelParser

# ``flat``: ``DataConfig.labels`` is a single ``LabelsConfig`` (not a dict of
# named entries), so the composed variant lands directly at ``cfg.data.labels``
# (package ``data.labels``), with parser names competing for that one slot.
LABELS = FamilyConfig(
    group="data/labels",
    schema=LabelsConfig,
    package_style="flat",
)

T = TypeVar("T", bound="LabelParser")


def label_parser(
    **common: Unpack[AdapterDecoratorOptions],
) -> Callable[[type[T]], type[T]]:
    """Decorator: register a label-parser adapter.

    ``registry_name`` is required. Mirrors ``metrics_adapter`` shape.

    Label parsers ship with the core install (they use already-required deps
    like pandas / Pillow, not an optional extra), so they declare no uv extra.
    Without this default the schema-backed auto-extra (``extra=registry_name``)
    would register phantom extras like ``tabular`` that no ``pyproject`` group
    provides, breaking the deps static-scan gate.
    """
    common.setdefault("extra", "")

    def wrap(cls: type[T]) -> type[T]:
        return _register_core(cls, family=LABELS, **common)

    return wrap
