from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResolvedReportSample:
    sample_index: int
    sample_id: str | None
    requested_sample: Any


def resolve_report_sample_selection(
    selection: Any,
    *,
    sample_ids: list[str] | None,
    batch_size: int,
) -> list[ResolvedReportSample] | None:
    """Resolve explicit report sample config to batch indices."""
    if selection is None:
        return None

    entries = _normalise_selection_entries(selection)
    ids = [] if sample_ids is None else [str(sample_id) for sample_id in sample_ids]
    resolved: list[ResolvedReportSample] = []
    seen: dict[int, Any] = {}

    for entry in entries:
        sample_index = _resolve_entry(entry, sample_ids=ids, batch_size=batch_size)
        if sample_index in seen:
            raise ValueError(
                "Duplicate report sample selection: "
                f"{entry!r} resolves to sample index {sample_index}, "
                f"already selected by {seen[sample_index]!r}."
            )
        seen[sample_index] = entry
        resolved.append(
            ResolvedReportSample(
                sample_index=sample_index,
                sample_id=ids[sample_index] if sample_index < len(ids) else None,
                requested_sample=entry,
            )
        )

    return resolved


def _normalise_selection_entries(selection: Any) -> list[Any]:
    if isinstance(selection, (str, bytes, Mapping)):
        raise ValueError(_selection_shape_error())
    try:
        entries = list(selection)
    except TypeError as error:
        raise ValueError(_selection_shape_error()) from error
    if not entries:
        raise ValueError(
            "reporting.sample_selection must contain at least one sample. "
            "Use null to keep automatic report sample selection."
        )
    return entries


def _selection_shape_error() -> str:
    return (
        "reporting.sample_selection must be a list of sample IDs, filenames, "
        "or zero-based indices. For one sample, use reporting.sample_selection=[...]."
    )


def _resolve_entry(entry: Any, *, sample_ids: list[str], batch_size: int) -> int:
    if isinstance(entry, bool):
        raise ValueError(f"Unsupported report sample selection entry {entry!r}.")
    if isinstance(entry, int):
        if entry < 0 or entry >= batch_size:
            raise ValueError(
                f"Report sample selection index {entry!r} is out of range; "
                f"valid range is {_valid_index_range(batch_size)}."
            )
        return entry
    if isinstance(entry, str):
        return _resolve_string_entry(entry, sample_ids=sample_ids, batch_size=batch_size)
    raise ValueError(f"Unsupported report sample selection entry {entry!r}.")


def _resolve_string_entry(entry: str, *, sample_ids: list[str], batch_size: int) -> int:
    if not sample_ids:
        raise ValueError(
            f"Report sample selection {entry!r} cannot be resolved because sample IDs "
            f"are unavailable; use an integer index in range {_valid_index_range(batch_size)}."
        )

    request = entry.strip().replace("\\", "/")
    request_without_suffix = _without_suffix(request)
    request_stem = Path(request).stem
    tiers = (
        (request, lambda sample_id: sample_id),
        (request_without_suffix, _without_suffix),
        (request_stem, lambda sample_id: Path(sample_id).stem),
    )

    for wanted, key_fn in tiers:
        matches = [
            index for index, sample_id in enumerate(sample_ids) if key_fn(sample_id) == wanted
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            matched_ids = ", ".join(sample_ids[index] for index in matches[:5])
            suffix = "" if len(matches) <= 5 else f", ... ({len(matches)} matches)"
            raise ValueError(
                f"Report sample selection {entry!r} is ambiguous; it matches {matched_ids}{suffix}."
            )

    preview = ", ".join(sample_ids[:5])
    suffix = "" if len(sample_ids) <= 5 else f", ... ({len(sample_ids)} total)"
    raise ValueError(
        f"Report sample selection {entry!r} did not match any sample ID. "
        f"Available sample IDs include: {preview}{suffix}. "
        f"Integer indices may use range {_valid_index_range(batch_size)}."
    )


def _without_suffix(value: str) -> str:
    return Path(value).with_suffix("").as_posix()


def _valid_index_range(batch_size: int) -> str:
    if batch_size <= 0:
        return "empty batch"
    return f"0..{batch_size - 1}"
