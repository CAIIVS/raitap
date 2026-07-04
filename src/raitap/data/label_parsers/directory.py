"""Directory label parser (torchvision ImageFolder semantics)."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from raitap import raitap_log
from raitap.configs.schema import DirectoryLabelsConfig
from raitap.data.label_parsers.registration import label_parser
from raitap.types import TaskKind
from raitap.utils.lazy import lazy_import

torch = lazy_import("torch")


@label_parser(registry_name="directory", schema=DirectoryLabelsConfig)
class DirectoryLabelParser:
    """Parse classification labels from the top-level class subfolder of each sample.

    Mirrors torchvision ``ImageFolder`` semantics: ``<class>/<file>`` layout.
    Uses ``sample_ids`` only; ignores ``data_source`` and ``class_names``.
    """

    supported_tasks: frozenset[TaskKind] = frozenset({TaskKind.classification})

    def parse(
        self,
        *,
        task_kind: TaskKind,
        tensor: Any,
        sample_ids: list[str] | None,
        data_source: str | None,
        class_names: list[str] | None,
    ) -> Any:
        """Derive a long-tensor of class indices from sample_ids directory layout."""
        if not sample_ids:
            raitap_log.warn(
                "DirectoryLabelParser needs image samples organised into "
                "class subdirectories; none were found. Falling back to "
                "predictions as metric targets."
            )
            return None
        parts_by_id = [PurePosixPath(sid).parts for sid in sample_ids]
        if any(len(parts) < 2 for parts in parts_by_id):
            raitap_log.warn(
                "DirectoryLabelParser expects a <class>/<file> layout, but "
                "one or more samples sit directly under the data source root "
                "(no class subdirectory). Falling back to predictions as metric targets."
            )
            return None
        classes = sorted({parts[0] for parts in parts_by_id})
        class_to_idx = {name: idx for idx, name in enumerate(classes)}
        labels = [class_to_idx[parts[0]] for parts in parts_by_id]
        return torch.tensor(labels, dtype=torch.long)
