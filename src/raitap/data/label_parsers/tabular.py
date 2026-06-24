"""Tabular label parser (CSV / TSV / Parquet)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap import raitap_log
from raitap.configs.schema import TabularLabelsConfig
from raitap.data.data import (
    SourceKind,
    _align_labels_to_samples,
    _column_as_series,
    _extract_class_labels,
    _load_tabular_frame,
    _resolve_id_strategy,
    _resolve_labels_id_column,
    get_source_path,
)
from raitap.data.label_parsers.registration import label_parser
from raitap.data.types import IdStrategy
from raitap.types import TaskKind
from raitap.utils.lazy import lazy_import

if TYPE_CHECKING:
    import torch

torch = lazy_import("torch")  # type: ignore[assignment]


@label_parser(registry_name="tabular", schema=TabularLabelsConfig)
class TabularLabelParser:
    """Parse classification labels from a CSV, TSV, or Parquet file.

    Aligns to ``sample_ids`` via ``id_column`` when available; falls back to
    row order otherwise. Returns ``None`` on empty file or count mismatch.
    """

    supported_tasks: frozenset[TaskKind] = frozenset({TaskKind.classification})

    def __init__(
        self,
        *,
        source: str,
        id_column: str | None = None,
        column: str | None = None,
        encoding: Any = None,
        id_strategy: IdStrategy = IdStrategy.auto,
    ) -> None:
        self.source = source
        self.id_column = id_column
        self.column = column
        self.encoding = encoding
        self.id_strategy = id_strategy

    def parse(
        self,
        *,
        task_kind: TaskKind,
        tensor: Any,
        sample_ids: list[str] | None,
        data_source: str | None,
        class_names: list[str] | None,
    ) -> Any:
        """Load tabular labels and align to samples."""
        labels_path = get_source_path(self.source, kind=SourceKind.LABELS)
        labels_df = _load_tabular_frame(labels_path)
        if labels_df.empty:
            raitap_log.warn("Labels file is empty; falling back to predictions as targets.")
            return None

        id_column = _resolve_labels_id_column(labels_df, self.id_column)
        encoded_labels = _extract_class_labels(
            labels_df,
            labels_column=self.column,
            id_column=id_column,
            labels_encoding=self.encoding,
        )

        expected = len(tensor) if tensor is not None else len(encoded_labels)
        if sample_ids and id_column:
            id_series = _column_as_series(labels_df, id_column)
            strategy = _resolve_id_strategy(str(self.id_strategy), id_series)
            try:
                aligned_labels = _align_labels_to_samples(
                    sample_ids=sample_ids,
                    raw_label_ids=id_series,
                    encoded_labels=encoded_labels,
                    strategy=strategy,
                )
            except ValueError as error:
                raitap_log.warn(
                    f"{error} Falling back to predictions as metric targets.",
                )
                return None
            return torch.tensor(aligned_labels, dtype=torch.long)

        if sample_ids and not id_column:
            raitap_log.warn(
                "Could not find a labels id column for filename alignment; using row-order labels.",
            )

        if len(encoded_labels) != expected:
            raitap_log.warn(
                f"Label count ({len(encoded_labels)}) does not match sample count ({expected}); "
                "falling back to predictions as targets.",
            )
            return None

        return torch.tensor(encoded_labels, dtype=torch.long)
