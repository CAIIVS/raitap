"""Guard: deps bootstrap must compose + infer extras with no torch backend.

This is the load-bearing test for the bootstrap-from-zero flow advertised in
``docs/using-raitap/get-it-running.md``. A user who runs ``pip install raitap``
(no extras) and then ``raitap --demo`` lands in :func:`_compose` with no torch
installed; if any module reachable from ``register_zen_groups()`` imports
``torch`` / ``torchvision`` / ``torch.nn`` at the top level, compose dies with
``ModuleNotFoundError`` BEFORE the bootstrap gets a chance to install the
right backend extra.

The per-family ``test_partial_extras_safe.py`` files cover the same contract
adapter-by-adapter; this file adds the bootstrap call path itself so a
regression surfaces in the deps test suite (where it belongs) rather than only
in one family's test.

Both tests are needed: the family tests pinpoint the offending module on
regression, this test confirms the integrated bootstrap path stays viable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

# Every backend library the deps system can install. Poisoning all of them
# matches the bootstrap-from-zero venv state (bare ``raitap`` install, no
# extras). Sub-modules listed explicitly because ``sys.modules`` poisoning
# only blocks future imports of the exact key.
_BACKEND_LIBS = (
    "torch",
    "torch.nn",
    "torchvision",
    "torchvision.models",
    "torchvision.transforms",
    "torchvision.transforms.v2",
    "torchvision.transforms._presets",
    "onnxruntime",
    "intel_extension_for_pytorch",
)


@pytest.fixture
def _hide_backend_libs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for lib in _BACKEND_LIBS:
        monkeypatch.setitem(sys.modules, lib, None)
    yield


def test_register_configs_without_backends(_hide_backend_libs: None) -> None:
    """``register_configs`` walks the family ``__init__``s — must stay
    torch-free so the bootstrap can compose before any backend is installed."""
    from raitap.configs import register_configs

    register_configs()


def test_compose_demo_config_without_backends(_hide_backend_libs: None) -> None:
    """The bundled ``demo.yaml`` is what ``raitap --demo`` runs first. If
    compose fails on a bare venv, the auto-deps frame the user sees says
    "No module named 'torch'" and the install never starts."""
    from raitap.deps.bootstrap import _compose

    demo_dir = Path(__file__).resolve().parents[2] / "configs"
    composed = _compose(demo_dir, "demo", [])
    assert isinstance(composed, dict)


def test_infer_extras_without_backends(_hide_backend_libs: None) -> None:
    """Bootstrap path: ``_compose`` → ``infer_extras``. The host probe picks
    the torch backend (``torch-cpu`` on the GH Linux runner / dev laptops
    without CUDA/XPU) and that extra must land in the result so the
    subsequent ``uv add raitap[...]`` installs torch."""
    from raitap.deps.bootstrap import _compose
    from raitap.deps.inference import infer_extras
    from raitap.types import ResolvedHardware

    demo_dir = Path(__file__).resolve().parents[2] / "configs"
    composed = _compose(demo_dir, "demo", [])
    extras, _ = infer_extras(composed, hardware=ResolvedHardware.cpu)

    # The demo config drives a torch model + transparency + metrics —
    # everything below is what makes the bootstrap-from-zero promise hold.
    assert "torch-cpu" in extras
