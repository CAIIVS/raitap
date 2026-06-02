"""Report sample-selection model.

Extracted from ``builder.py`` so per-phase report renderers and the builder
share one definition of which samples a report highlights. Selection keys off
``RunOutputs.prediction_summaries`` / ``forward_output`` (top-level, cross-phase
fields), so the strategies stay phase-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from raitap.pipeline.outputs import PredictionSummary

if TYPE_CHECKING:
    from raitap.pipeline.outputs import RunOutputs
    from raitap.reporting.sample_selection import ResolvedReportSample

# Caps both the selected sample pool and, after reserving the first item for the
# overview, the number of detail groups shown in the local section.
_MAX_LOCAL_DETAIL_SAMPLES = 3


class SelectionSource(StrEnum):
    AUTOMATIC = "automatic"
    USER = "user"


@dataclass(frozen=True)
class SelectedSample:
    label: str
    summary: PredictionSummary
    selection_source: SelectionSource = SelectionSource.AUTOMATIC
    requested_sample: object | None = None


def report_batch_size(outputs: RunOutputs) -> int:
    return outputs.forward_output.batch_size


def _requested_sample_metadata(sample: SelectedSample) -> dict[str, object]:
    if sample.requested_sample is None:
        return {}
    return {"requested_sample": sample.requested_sample}


class SampleSelectionStrategy(ABC):
    @abstractmethod
    def select(self, outputs: RunOutputs) -> list[SelectedSample]:
        """Return the samples that should be highlighted in the local section."""


class UserSelectorStrategy(SampleSelectionStrategy):
    def __init__(self, resolved_samples: list[ResolvedReportSample]) -> None:
        self._resolved_samples = resolved_samples

    def select(self, outputs: RunOutputs) -> list[SelectedSample]:
        summaries = {summary.sample_index: summary for summary in outputs.prediction_summaries}
        return [
            SelectedSample(
                label="user_selected",
                summary=summaries.get(
                    resolved.sample_index,
                    PredictionSummary(
                        sample_index=resolved.sample_index,
                        predicted_class=-1,
                        confidence=0.0,
                        sample_id=resolved.sample_id,
                    ),
                ),
                selection_source=SelectionSource.USER,
                requested_sample=resolved.requested_sample,
            )
            for resolved in self._resolved_samples
        ]


class EdgecaseSelectorStrategy(SampleSelectionStrategy):
    def select(self, outputs: RunOutputs) -> list[SelectedSample]:
        summaries = list(outputs.prediction_summaries)
        if summaries:
            picked = self._pick_tiered(summaries)
            if picked:
                return picked
        return self._placeholder_samples(outputs)

    def _pick_tiered(self, summaries: list[PredictionSummary]) -> list[SelectedSample]:
        selected: list[SelectedSample] = []
        seen: set[int] = set()
        tiers = (
            ("wrong", [s for s in summaries if s.correct is False], True),
            ("insecure", summaries, False),
            ("high_confidence", [s for s in summaries if s.correct is not False], True),
        )
        for label, candidates, reverse in tiers:
            if not candidates:
                continue
            ordered = sorted(candidates, key=lambda item: item.confidence, reverse=reverse)
            for candidate in ordered:
                if candidate.sample_index in seen:
                    continue
                selected.append(SelectedSample(label=label, summary=candidate))
                seen.add(candidate.sample_index)
                break
            if len(selected) >= _MAX_LOCAL_DETAIL_SAMPLES:
                break
        return selected

    def _placeholder_samples(self, outputs: RunOutputs) -> list[SelectedSample]:
        total = report_batch_size(outputs)
        if total <= 0:
            return []
        count = min(total, _MAX_LOCAL_DETAIL_SAMPLES)
        sample_ids = outputs.sample_ids or []
        return [
            SelectedSample(
                label="sample",
                summary=PredictionSummary(
                    sample_index=index,
                    predicted_class=-1,
                    confidence=0.0,
                    sample_id=sample_ids[index] if index < len(sample_ids) else None,
                ),
            )
            for index in range(count)
        ]
