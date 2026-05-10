from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .sections import ReportGroup, ReportSection


_ROBUSTNESS_IMAGE_RE = re.compile(
    r"robustness_(?P<result_index>\d+)_(?P<assessor>.+)_sample_"
    r"(?P<sample_index>\d+)_(?P<visualiser>.+)\.png$"
)
_HEADLINE_KEYS = frozenset({"explainer", "algorithm", "method_families", "visualiser_title"})
_CONTEXT_KEYS = frozenset(
    {"targets", "visualiser_method", "visualiser_sign", "output_space", "layer_path"}
)
_TECHNICAL_KEYS = frozenset(
    {
        "output_layout",
        "output_shape",
        "requires_interpolation",
        "visualiser_class",
    }
)
_ROBUSTNESS_SETTING_KEYS = frozenset(
    {"assessor", "algorithm", "method_kind", "threat_model", "objective", "norm", "epsilon"}
)


@dataclass(frozen=True, slots=True)
class MetricsView:
    heading: str
    rows: tuple[tuple[str, str], ...]
    image_srcs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExplainerView:
    explainer_name: str
    algorithm: str
    visualiser_name: str
    image_srcs: tuple[str, ...]
    rows: tuple[tuple[str, str], ...]
    headline: dict[str, str]
    context: dict[str, str]
    technical: dict[str, str]


@dataclass(frozen=True, slots=True)
class LocalSampleView:
    sample_index: int
    bucket: str
    heading: str
    rows: tuple[tuple[str, str], ...]
    thumbnail_srcs: tuple[str, ...]
    explainers: tuple[ExplainerView, ...] = ()
    sample_id: str | None = None
    predicted_class: str | None = None
    confidence: str | None = None
    target_class: str | None = None
    correct: bool | None = None


@dataclass(frozen=True, slots=True)
class RobustnessSampleEvidence:
    sample_index: int
    visualisers: dict[str, str]

    @property
    def image_pair_src(self) -> str | None:
        return self.visualisers.get("ImagePairVisualiser")

    @property
    def perturbation_heatmap_src(self) -> str | None:
        return self.visualisers.get("PerturbationHeatmapVisualiser")


@dataclass(frozen=True, slots=True)
class RobustnessAssessorView:
    assessor_name: str
    algorithm: str
    method_kind: str
    heading: str
    rows: tuple[tuple[str, str], ...]
    settings: dict[str, str]
    metrics: dict[str, str]
    samples: tuple[RobustnessSampleEvidence, ...]


@dataclass(frozen=True, slots=True)
class SummaryView:
    experiment_name: str
    sample_count: int
    explainer_count: int
    assessor_count: int
    metric_highlights: tuple[tuple[str, str], ...] = ()
    robustness_highlights: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class AppendixView:
    sections: tuple[ReportSection, ...]


@dataclass(frozen=True, slots=True)
class GenericGroupView:
    heading: str
    rows: tuple[tuple[str, str], ...]
    image_srcs: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReportView:
    experiment_name: str
    generated_at: str
    version: str
    summary: SummaryView
    metrics: MetricsView | None
    global_groups: tuple[GenericGroupView, ...]
    cohort_groups: tuple[GenericGroupView, ...]
    local_samples: tuple[LocalSampleView, ...]
    robustness_assessors: tuple[RobustnessAssessorView, ...]
    appendix: AppendixView


def build_view(
    sections: Sequence[ReportSection],
    metadata: Mapping[str, Any] | None = None,
    *,
    version: str = "0+unknown",
    generated_at: datetime | None = None,
) -> ReportView:
    """Map report sections into a template-friendly view model."""
    metadata = {} if metadata is None else metadata
    metrics: MetricsView | None = None
    global_groups: tuple[GenericGroupView, ...] = ()
    cohort_groups: tuple[GenericGroupView, ...] = ()
    local_samples: tuple[LocalSampleView, ...] = ()
    robustness_assessors: tuple[RobustnessAssessorView, ...] = ()

    for section in sections:
        role = _section_role(section)
        if role == "metrics":
            metrics = _build_metrics_view(section)
        elif role == "global":
            global_groups = _build_generic_groups(section)
        elif role == "cohort":
            cohort_groups = _build_generic_groups(section)
        elif role == "local":
            local_samples = _build_local_samples(section)
        elif role == "robustness":
            robustness_assessors = _build_robustness_assessors(section)

    experiment_name = _none_if_blank(metadata.get("experiment_name")) or "n/a"
    summary = SummaryView(
        experiment_name=experiment_name,
        sample_count=len(local_samples),
        explainer_count=len(
            {
                explainer.explainer_name
                for sample in local_samples
                for explainer in sample.explainers
            }
        ),
        assessor_count=len(robustness_assessors),
        metric_highlights=metrics.rows if metrics is not None else (),
        robustness_highlights=tuple(
            (
                assessor.assessor_name,
                assessor.metrics.get("clean_accuracy", "n/a"),
                assessor.metrics.get("adversarial_accuracy", "n/a"),
            )
            for assessor in robustness_assessors
        ),
    )
    generated = generated_at if generated_at is not None else datetime.now().astimezone()

    return ReportView(
        experiment_name=experiment_name,
        generated_at=generated.isoformat(timespec="seconds"),
        version=version,
        summary=summary,
        metrics=metrics,
        global_groups=global_groups,
        cohort_groups=cohort_groups,
        local_samples=local_samples,
        robustness_assessors=robustness_assessors,
        appendix=AppendixView(sections=tuple(sections)),
    )


def _section_role(section: ReportSection) -> str:
    raw = str(section.metadata.get("section_role", ""))
    aliases = {
        "global_explanations": "global",
        "cohort_explanations": "cohort",
        "local_explanations": "local",
    }
    return aliases.get(raw, raw)


def _build_metrics_view(section: ReportSection) -> MetricsView | None:
    for group in section.groups:
        if group.table_rows or group.images:
            return MetricsView(
                heading=group.heading,
                rows=group.table_rows,
                image_srcs=_image_srcs(group),
            )
    return None


def _build_generic_groups(section: ReportSection) -> tuple[GenericGroupView, ...]:
    return tuple(
        GenericGroupView(
            heading=group.heading,
            rows=group.table_rows,
            image_srcs=_image_srcs(group),
            metadata=dict(group.metadata),
        )
        for group in section.groups
        if group.table_rows or group.images
    )


def _build_local_samples(section: ReportSection) -> tuple[LocalSampleView, ...]:
    samples_by_index: dict[int, LocalSampleView] = {}
    explainers_by_index: dict[int, list[ExplainerView]] = {}

    for group in section.groups:
        role = group.metadata.get("role")
        if role not in {"sample_header", "local_visualiser"}:
            continue
        sample_index = _metadata_int(group.metadata, "sample_index")
        if sample_index is None:
            continue

        if role == "sample_header":
            rows = _as_dict(group.table_rows)
            samples_by_index[sample_index] = LocalSampleView(
                sample_index=sample_index,
                bucket=str(group.metadata.get("bucket") or rows.get("bucket") or "sample"),
                heading=group.heading,
                rows=group.table_rows,
                thumbnail_srcs=_image_srcs(group),
                explainers=(),
                sample_id=rows.get("sample_id"),
                predicted_class=rows.get("predicted_class"),
                confidence=rows.get("confidence"),
                target_class=rows.get("target_class"),
                correct=_parse_bool(rows.get("correct")),
            )
            explainers_by_index.setdefault(sample_index, [])
            continue

        explainer = _build_explainer_view(group)
        explainers_by_index.setdefault(sample_index, []).append(explainer)
        if sample_index not in samples_by_index:
            rows = _as_dict(group.table_rows)
            samples_by_index[sample_index] = LocalSampleView(
                sample_index=sample_index,
                bucket=str(group.metadata.get("bucket") or rows.get("bucket") or "sample"),
                heading=f"Sample {sample_index}",
                rows=(),
                thumbnail_srcs=(),
            )

    ordered: list[LocalSampleView] = []
    for sample_index in samples_by_index:
        sample = samples_by_index[sample_index]
        ordered.append(
            LocalSampleView(
                sample_index=sample.sample_index,
                bucket=sample.bucket,
                heading=sample.heading,
                rows=sample.rows,
                thumbnail_srcs=sample.thumbnail_srcs,
                explainers=tuple(explainers_by_index.get(sample_index, [])),
                sample_id=sample.sample_id,
                predicted_class=sample.predicted_class,
                confidence=sample.confidence,
                target_class=sample.target_class,
                correct=sample.correct,
            )
        )
    return tuple(ordered)


def _build_explainer_view(group: ReportGroup) -> ExplainerView:
    rows = group.table_rows
    row_map = _as_dict(rows)
    headline: dict[str, str] = {}
    context: dict[str, str] = {}
    technical: dict[str, str] = {}

    for key, value in rows:
        if key in _HEADLINE_KEYS:
            headline[key] = value
        elif key in _CONTEXT_KEYS:
            context[key] = value
        elif key.startswith(("input_", "call.", "visualiser_call.")) or key in _TECHNICAL_KEYS:
            technical[key] = value
        else:
            technical[key] = value

    explainer_name = str(group.metadata.get("explainer_name") or row_map.get("explainer") or "n/a")
    algorithm = str(group.metadata.get("algorithm") or row_map.get("algorithm") or "n/a")
    visualiser_name = str(
        group.metadata.get("visualiser_name")
        or group.metadata.get("visualiser_title")
        or row_map.get("visualiser_title")
        or "visualiser"
    )

    return ExplainerView(
        explainer_name=explainer_name,
        algorithm=algorithm,
        visualiser_name=visualiser_name,
        image_srcs=_image_srcs(group),
        rows=rows,
        headline=headline,
        context=context,
        technical=technical,
    )


def _build_robustness_assessors(section: ReportSection) -> tuple[RobustnessAssessorView, ...]:
    assessors: list[RobustnessAssessorView] = []
    for group in section.groups:
        if group.metadata.get("role") != "robustness":
            continue
        rows = _as_dict(group.table_rows)
        settings = {
            key: value for key, value in group.table_rows if key in _ROBUSTNESS_SETTING_KEYS
        }
        metrics = {
            key: value for key, value in group.table_rows if key not in _ROBUSTNESS_SETTING_KEYS
        }
        assessor_name = str(group.metadata.get("assessor_name") or rows.get("assessor") or "n/a")
        sample_order = _robustness_sample_order(group)
        evidence_by_index = _robustness_evidence_by_index(group)
        assessors.append(
            RobustnessAssessorView(
                assessor_name=assessor_name,
                algorithm=str(group.metadata.get("algorithm") or rows.get("algorithm") or "n/a"),
                method_kind=str(
                    group.metadata.get("method_kind") or rows.get("method_kind") or "n/a"
                ),
                heading=group.heading,
                rows=group.table_rows,
                settings=settings,
                metrics=metrics,
                samples=tuple(
                    RobustnessSampleEvidence(
                        sample_index=sample_index,
                        visualisers=evidence_by_index.get(sample_index, {}),
                    )
                    for sample_index in sample_order
                ),
            )
        )
    return tuple(assessors)


def _robustness_sample_order(group: ReportGroup) -> tuple[int, ...]:
    raw = group.metadata.get("sample_indices")
    if isinstance(raw, list | tuple):
        ordered = tuple(converted for value in raw if (converted := _to_int(value)) is not None)
        if ordered:
            return ordered
    return tuple(_robustness_evidence_by_index(group))


def _robustness_evidence_by_index(group: ReportGroup) -> dict[int, dict[str, str]]:
    evidence: dict[int, dict[str, str]] = {}
    for image in group.images:
        match = _ROBUSTNESS_IMAGE_RE.search(Path(image).name)
        if match is None:
            continue
        sample_index = int(match.group("sample_index"))
        visualiser = _strip_visualiser_index(match.group("visualiser"))
        evidence.setdefault(sample_index, {})[visualiser] = _image_src(image)
    return evidence


def _strip_visualiser_index(value: str) -> str:
    return re.sub(r"_\d+$", "", value)


def _metadata_int(metadata: Mapping[str, object], key: str) -> int | None:
    return _to_int(metadata.get(key))


def _to_int(value: object) -> int | None:
    if not isinstance(value, str | int):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_dict(rows: tuple[tuple[str, str], ...]) -> dict[str, str]:
    return dict(rows)


def _image_srcs(group: ReportGroup) -> tuple[str, ...]:
    return tuple(_image_src(image) for image in group.images)


def _image_src(path: Path) -> str:
    parts = Path(path).parts
    if "_assets" in parts:
        index = parts.index("_assets")
        return Path(*parts[index:]).as_posix()
    return Path(path).as_posix()


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def _none_if_blank(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
