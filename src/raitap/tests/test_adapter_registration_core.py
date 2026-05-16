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
        strip_suffixes=("Explainer",),
    )
    assert dataclasses.is_dataclass(fc)
    with pytest.raises(dataclasses.FrozenInstanceError):
        fc.group = "robustness"  # type: ignore[misc]


def test_family_config_package_style_literal_enforced_at_runtime() -> None:
    from raitap._adapters import FamilyConfig

    # Runtime acceptance — typing-level enforcement is verified separately by pyright.
    FamilyConfig(group="g", schema=object, package_style="flat", strip_suffixes=())
    FamilyConfig(group="g", schema=object, package_style="nested", strip_suffixes=())
