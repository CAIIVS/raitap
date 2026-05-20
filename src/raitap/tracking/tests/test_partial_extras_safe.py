"""Guard: every adapter module in ``raitap.tracking`` must import cleanly
when its wrapped third-party libraries are not installed.

See ``raitap.metrics.tests.test_partial_extras_safe`` for the rationale.
"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest

_WRAPPED_LIBS = ("torch", "torch.nn", "torchvision", "mlflow")

_ADAPTER_MODULES = (
    "raitap.tracking",
    "raitap.tracking.mlflow_tracker",
    "raitap.tracking.base_tracker",
)


@pytest.fixture
def _hide_wrapped_libs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
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


def test_facades_import_without_torch() -> None:
    for mod in ("raitap.adapters", "raitap.visualisers", "raitap.backends"):
        importlib.import_module(mod)
