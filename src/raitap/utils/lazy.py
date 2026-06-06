"""Lazy-import proxy for adapter-wrapped third-party libraries.

Adapter modules can be imported in partial-extras venvs (e.g. while
:func:`raitap.run` with ``auto_install_deps=True`` is still figuring out which
``uv add`` to run, or during the bootstrap-from-zero ``raitap --demo`` flow
when no torch backend is installed yet) only if they do not pull their
wrapped library at module-load time. The proxy here defers the real import
until first attribute access — when the code inside an adapter method
actually needs the library — so::

    from raitap.utils.lazy import lazy_import
    torchmetrics = lazy_import("torchmetrics")

    class MulticlassClassificationMetrics(...):
        def __init__(self, *, num_classes: int) -> None:
            self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

…lets ``from raitap.metrics.classification_metrics import
MulticlassClassificationMetrics`` succeed in a venv that has no ``torchmetrics``
installed, as long as nobody actually constructs the class.

Pair with a ``TYPE_CHECKING`` import so Pyright still sees the real
module's types::

    if TYPE_CHECKING:
        import torchmetrics
    else:
        torchmetrics = lazy_import("torchmetrics")

**Scope: backend libraries too.** The contract applies to ``torch``,
``torchvision``, ``onnxruntime``, ``intel_extension_for_pytorch`` —
every library the deps-bootstrap can install — not just the optional
extras. Without this, ``raitap.deps.bootstrap._compose`` (which imports
every adapter family ``__init__`` to walk ``_target_``s) dies on a bare
venv and the bootstrap never gets to install the right backend.

**Class definitions that subclass a wrapped lib.** ``class Foo(nn.Module):``
evaluates ``nn.Module`` at class-def time → forces a real import → breaks
the contract. Wrap such classes in a lazy factory function::

    _FOO_CLS: type | None = None

    def _foo_cls() -> type:
        global _FOO_CLS
        if _FOO_CLS is not None:
            return _FOO_CLS
        class Foo(nn.Module):
            ...
        _FOO_CLS = Foo
        return Foo

…and call ``_foo_cls()(...)`` at construction sites. See
``raitap.data.preprocessing._preset_wrapper_cls`` for a production example.

**``isinstance`` checks against wrapped-lib classes.** ``isinstance(x, FooClass)``
needs ``FooClass`` to be a real type — lazy proxies fail
``__instancecheck__``. Resolve the type via attribute access at the call
site: bind the parent module lazily and write
``isinstance(x, _module.FooClass)``. See
``raitap.data.preprocessing`` for the ``_presets`` pattern.

The hand-maintained guard is intentional: a CI test in each adapter
family (``<family>/tests/test_partial_extras_safe.py``) re-imports every
adapter with the wrapped libraries shadowed in ``sys.modules``, and
``raitap.deps.tests.test_bootstrap_from_zero`` covers the bootstrap call
path with all backend libs poisoned. A maintainer who forgets the
pattern and adds a top-level ``import torchmetrics`` (or
``import torch``) fails immediately.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType


class _LazyModule:
    """Proxy that imports its backing module on first attribute access.

    Attribute writes (``monkeypatch.setattr(proxy, "x", ...)``) are forwarded
    to the real module so test patches behave the same as on a regular import.
    Internal slots (``_mod``, ``_name``) keep the indirection cheap.
    """

    __slots__ = ("_mod", "_name")

    def __init__(self, name: str) -> None:
        # Bypass our own __setattr__ during init — the slot machinery needs
        # raw assignments before ``_mod`` exists to dispatch through.
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_mod", None)

    def _load(self) -> ModuleType:
        mod = self._mod
        if mod is None:
            mod = importlib.import_module(self._name)
            object.__setattr__(self, "_mod", mod)
        return mod

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._load(), attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr in ("_mod", "_name"):
            object.__setattr__(self, attr, value)
            return
        setattr(self._load(), attr, value)

    def __delattr__(self, attr: str) -> None:
        if attr in ("_mod", "_name"):
            object.__delattr__(self, attr)
            return
        delattr(self._load(), attr)

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
