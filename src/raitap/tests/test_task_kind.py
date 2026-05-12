"""Tests for the shared TaskKind taxonomy and supported_tasks ClassVar."""

from __future__ import annotations

import pytest

from raitap.semantics_base import SemanticallyDescribable, TaskKind


def test_task_kind_members():
    assert {member.value for member in TaskKind} == {
        "classification",
        "detection",
        "segmentation",
        "seq2seq",
        "regression",
    }


def test_supported_tasks_defaults_to_classification():
    class _Adapter(SemanticallyDescribable[str]):
        algorithm_registry = {"foo": "bar"}

    assert _Adapter.supported_tasks == frozenset({TaskKind.CLASSIFICATION})


def test_supported_tasks_can_be_overridden():
    class _DetectionAdapter(SemanticallyDescribable[str]):
        algorithm_registry = {"foo": "bar"}
        supported_tasks = frozenset({TaskKind.DETECTION})

    assert _DetectionAdapter.supported_tasks == frozenset({TaskKind.DETECTION})


def test_supported_tasks_must_be_frozenset_of_task_kind():
    with pytest.raises(TypeError, match="supported_tasks"):

        class _BadAdapter(SemanticallyDescribable[str]):
            algorithm_registry = {"foo": "bar"}
            supported_tasks = frozenset({"detection"})  # type: ignore[assignment]
