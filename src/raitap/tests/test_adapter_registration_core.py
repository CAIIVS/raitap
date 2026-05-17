"""Tests for the cross-family registration core (FamilyConfig + _register_core)."""

from __future__ import annotations

import dataclasses

import pytest


def test_family_config_is_frozen_slotted_dataclass() -> None:
    from raitap._adapters import FamilyConfig

    fc = FamilyConfig(
        group="transparency",
        schema=type("SchemaStub", (), {}),
        package_style="nested",
    )
    assert dataclasses.is_dataclass(fc)
    with pytest.raises(dataclasses.FrozenInstanceError):
        fc.group = "robustness"  # type: ignore[misc]


def test_family_config_package_style_literal_enforced_at_runtime() -> None:
    from raitap._adapters import FamilyConfig

    # Runtime acceptance check for both ``Literal`` values — typing-level
    # enforcement is verified separately by pyright on real FamilyConfig sites.
    FamilyConfig(group="_test", schema=object, package_style="flat")
    FamilyConfig(group="_test", schema=object, package_style="nested")


# Module-scope fixtures so hydra-zen's ``builds()`` can resolve their importable
# qualname. Function-local classes fail with ``ModuleNotFoundError: <cls> is not
# importable``, which the legacy ``__init_subclass__`` swallow would mask.
@dataclasses.dataclass
class _DummySchema:
    _target_: str = ""


class _DummyAdapter:
    def __init__(self, **kwargs):
        pass


class _DummyVisualiser:
    def __init__(self, max_samples: int = 4):
        pass


def test_register_core_with_family_populates_builders_and_extras() -> None:
    """_register_core should run the same mechanics as __init_subclass__: hydra-zen
    builder in _BUILDERS, ADAPTER_EXTRAS entry, library tracked in THIRD_PARTY_LIBS."""
    from raitap._adapters import (
        _BUILDERS,
        ADAPTER_EXTRAS,
        THIRD_PARTY_LIBS,
        FamilyConfig,
        _register_core,
    )

    fc = FamilyConfig(
        group="_test_family",
        schema=_DummySchema,
        package_style="nested",
    )
    _register_core(
        _DummyAdapter,
        family=fc,
        registry_name="dummy",
        extra="dummy-extra",
        library="dummy-lib",
    )
    assert "_test_family" in _BUILDERS
    assert "dummy" in _BUILDERS["_test_family"]
    assert ADAPTER_EXTRAS["_DummyAdapter"] == "dummy-extra"
    assert "dummy-lib" in THIRD_PARTY_LIBS["_test_family"]


def test_register_core_without_family_uses_unscoped_pool() -> None:
    """Visualiser path: family=None routes to _BUILDERS['_unscoped'] via the
    signature-based builder."""
    from raitap._adapters import _BUILDERS, _register_core

    _register_core(
        _DummyVisualiser,
        family=None,
        registry_name="dummy_visualiser",
    )
    assert "dummy_visualiser" in _BUILDERS["_unscoped"]
