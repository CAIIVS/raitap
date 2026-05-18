"""Tests for the shared TaskKind taxonomy and AdapterMixin.supported_tasks default."""

from __future__ import annotations

from raitap._adapters import AdapterMixin
from raitap.types import TaskKind


def test_task_kind_members() -> None:
    assert {member.value for member in TaskKind} == {
        "classification",
        "detection",
        "segmentation",
        "seq2seq",
        "regression",
    }


def test_task_kind_string_round_trip() -> None:
    # StrEnum members are string subclasses; round-tripping through value
    # matches the OmegaConf convention used by Hardware / Task in types.py.
    assert TaskKind("detection") is TaskKind.detection
    assert TaskKind.detection == "detection"


def test_adapter_mixin_defaults_to_classification() -> None:
    assert AdapterMixin.supported_tasks == frozenset({TaskKind.classification})


def test_supported_tasks_can_be_overridden_on_subclass() -> None:
    class _DetectionAdapter(AdapterMixin):
        supported_tasks = frozenset({TaskKind.detection})

    assert _DetectionAdapter.supported_tasks == frozenset({TaskKind.detection})
    # Sibling subclass without an override keeps the AdapterMixin default,
    # confirming subclass overrides don't bleed via mutable state.

    class _PlainAdapter(AdapterMixin):
        pass

    assert _PlainAdapter.supported_tasks == frozenset({TaskKind.classification})
