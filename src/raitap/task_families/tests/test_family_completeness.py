from __future__ import annotations

from raitap.task_families.base import TaskFamily
from raitap.task_families.registry import TASK_FAMILIES


def test_all_registered_families_satisfy_protocol() -> None:
    assert TASK_FAMILIES  # non-empty
    for kind, fam in TASK_FAMILIES.items():
        assert isinstance(fam, TaskFamily), kind
        assert fam.kind is kind
