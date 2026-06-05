from __future__ import annotations

import pytest

from raitap.task_families.registry import (
    TASK_FAMILIES,
    resolve_task_family,
    task_family,
)
from raitap.types import TaskKind


def test_decorator_registers_under_kind() -> None:
    @task_family
    class _Fam:
        kind = TaskKind.regression
        output_space = None

    assert TASK_FAMILIES[TaskKind.regression] is not None
    assert resolve_task_family(TaskKind.regression).kind is TaskKind.regression
    # cleanup so the test is idempotent
    del TASK_FAMILIES[TaskKind.regression]


def test_resolve_unknown_kind_raises() -> None:
    with pytest.raises(KeyError):
        resolve_task_family(TaskKind.seq2seq)
