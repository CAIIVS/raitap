"""Lazy-import proxy for adapter-wrapped third-party libraries.

Adapter modules can be imported in partial-extras venvs (e.g. while
:func:`raitap.deps.install_raitap_deps` is still figuring out which
``uv add`` to run) only if they do not pull their wrapped library at
module-load time. The proxy here defers the real import until first
attribute access — when the code inside an adapter method actually
needs the library — so::

    from raitap.utils.lazy import lazy_import
    torchmetrics = lazy_import("torchmetrics")

    class ClassificationMetrics(...):
        def __init__(self) -> None:
            self.acc = torchmetrics.Accuracy(task="multiclass")

…lets ``from raitap.metrics.classification_metrics import
ClassificationMetrics`` succeed in a venv that has no ``torchmetrics``
installed, as long as nobody actually constructs the class.

Pair with a ``TYPE_CHECKING`` import so Pyright still sees the real
module's types::

    if TYPE_CHECKING:
        import torchmetrics
    else:
        torchmetrics = lazy_import("torchmetrics")

The hand-maintained guard is intentional: a CI test in each adapter
family (``<family>/tests/test_partial_extras_safe.py``) re-imports every
adapter with the wrapped libraries shadowed in ``sys.modules`` so a
maintainer who forgets the pattern and adds a top-level
``import torchmetrics`` fails immediately.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any


class _LazyModule:
    """Proxy that imports its backing module on first attribute access."""

    __slots__ = ("_mod", "_name")

    def __init__(self, name: str) -> None:
        self._name = name
        self._mod: ModuleType | None = None

    def __getattr__(self, attr: str) -> Any:
        if self._mod is None:
            self._mod = importlib.import_module(self._name)
        return getattr(self._mod, attr)

    def __repr__(self) -> str:
        status = "loaded" if self._mod is not None else "deferred"
        return f"<lazy_import {self._name!r} ({status})>"


def lazy_import(name: str) -> Any:
    """Return a proxy whose attribute access triggers ``import <name>``.

    The return type is ``Any`` rather than ``ModuleType`` so Pyright will
    not flag attribute accesses on the proxy — callers should pair this
    with a ``TYPE_CHECKING: import <name>`` block for real static typing.
    """
    return _LazyModule(name)
