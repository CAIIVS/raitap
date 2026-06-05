from raitap.task_families.base import ExplainContext, ForwardContext, TaskFamily
from raitap.task_families.registry import (
    TASK_FAMILIES,
    resolve_task_family,
    task_family,
)

__all__ = [
    "TASK_FAMILIES",
    "ExplainContext",
    "ForwardContext",
    "TaskFamily",
    "resolve_task_family",
    "task_family",
]

# Import the concrete families for their registration side effects, so that
# importing ``raitap.task_families`` populates ``TASK_FAMILIES``.
from . import classification as _classification  # pyright: ignore[reportUnusedImport]
from . import detection as _detection  # pyright: ignore[reportUnusedImport]
