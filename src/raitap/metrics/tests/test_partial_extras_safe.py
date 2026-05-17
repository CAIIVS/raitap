"""Guard: every adapter module in ``raitap.metrics`` must import cleanly when
its wrapped third-party libraries are not installed.

This is what unlocks the programmatic-API auto-deps flow — see
``raitap.run(..., auto_install_deps=True)`` in
``docs/using-raitap/configuration/python-api.md``. A maintainer who adds a
new adapter and forgets the :func:`raitap.utils.lazy.lazy_import` pattern
(adding a top-level ``import torchmetrics`` instead) breaks the
partial-extras-venv contract; this test fails immediately with the exact
module that regressed.
"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest

# Wrapped libraries that the ``metrics`` family of adapters consume *as
# optional extras*. Any top-level import of one of these from within a module
# under ``raitap.metrics`` is a contract violation — wrap it in
# :func:`raitap.utils.lazy.lazy_import` instead.
#
# ``torch`` + ``torchvision`` are also poisoned here even though every torch
# *backend* extra (torch-cpu/cuda/intel) provides them: the deps-bootstrap
# composes the Hydra config and walks adapter imports BEFORE installing any
# backend, so adapter modules must survive a torch-less venv long enough for
# the bootstrap to infer + install the right backend. See
# ``raitap.utils.lazy`` for the contract and ``raitap.deps.bootstrap`` for
# the call path.
_WRAPPED_LIBS = ("torch", "torch.nn", "torchvision", "torchmetrics")

# Adapter / adapter-adjacent modules that must survive the contract. Listed by
# hand (one entry per file) so adding a new adapter is a single-file change to
# the adapter plus a single-line addition here — no auto-discovery magic that
# could mask a missed registration.
_ADAPTER_MODULES = (
    "raitap.metrics",  # package ``__init__`` re-exports everything
    "raitap.metrics.classification_metrics",
    "raitap.metrics.detection_metrics",
    "raitap.metrics.factory",
    "raitap.metrics.visualizers",
)


@pytest.fixture
def _hide_wrapped_libs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Poison ``sys.modules`` so any wrapped-lib import raises ``ImportError``.

    Setting an entry to ``None`` makes ``importlib`` raise
    ``ModuleNotFoundError`` on the next ``import`` of that name — closer to a
    real partial-extras venv than ``del sys.modules[name]`` would be (the
    latter just forces a re-import from disk if the package is installed).
    """
    for lib in _WRAPPED_LIBS:
        monkeypatch.setitem(sys.modules, lib, None)
    yield


@pytest.mark.parametrize("module_name", _ADAPTER_MODULES)
def test_adapter_imports_without_wrapped_libs(
    module_name: str,
    _hide_wrapped_libs: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    importlib.import_module(module_name)
