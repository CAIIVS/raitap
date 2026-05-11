from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import torch

from raitap import raitap_log
from raitap.configs import resolve_run_dir
from raitap.robustness.contracts import MethodKind
from raitap.run.outputs import PredictionSummary, RunOutputs
from raitap.transparency.contracts import ExplanationScope, VisualisationContext
from raitap.transparency.visualisers import BaseVisualiser, InputThumbnailVisualiser

from .manifest import ReportManifest
from .sample_selection import ResolvedReportSample, resolve_report_sample_selection
from .sections import ReportGroup, ReportSection

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig
    from raitap.robustness.results import RobustnessVisualisationResult
    from raitap.robustness.visualisers.base_visualiser import BaseRobustnessVisualiser
    from raitap.transparency.results import VisualisationResult

# Caps both the selected sample pool and, after reserving the first item for the
# overview, the number of detail groups shown in the local section.
_MAX_LOCAL_DETAIL_SAMPLES = 3
_DISPLAY_ONLY_VISUALISER_KWARGS = frozenset(
    {
        "max_samples",
        "sample_index",
        "sample_names",
        "show_sample_names",
        "include_original_input",
        "include_original_image",
        "show_colorbar",
        "colorbar",
    }
)


class SelectionSource(StrEnum):
    AUTOMATIC = "automatic"
    USER = "user"


@dataclass(frozen=True)
class SelectedSample:
    label: str
    summary: PredictionSummary
    selection_source: SelectionSource = SelectionSource.AUTOMATIC
    requested_sample: object | None = None


@dataclass(frozen=True)
class BuiltReport:
    report_dir: Path
    sections: tuple[ReportSection, ...]
    manifest: ReportManifest


@dataclass(frozen=True)
class _StagedThumbnail:
    path: Path
    source_explainer_name: str


def build_report(config: AppConfig, outputs: RunOutputs) -> BuiltReport:
    report_dir = resolve_run_dir(config, subdir="reports")
    assets_dir = report_dir / "_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    reporting_cfg = config.reporting
    configured_selection = None if reporting_cfg is None else reporting_cfg.sample_selection
    explicit_samples = resolve_report_sample_selection(
        configured_selection,
        sample_ids=outputs.sample_ids,
        batch_size=_batch_size(outputs),
    )
    selected_samples = (
        _explicit_selected_samples(explicit_samples, outputs=outputs)
        if explicit_samples is not None
        else _select_samples(outputs)
    )
    sections: list[ReportSection] = []

    metrics_section = _build_metrics_section(outputs, assets_dir=assets_dir)
    if metrics_section is not None:
        sections.append(metrics_section)

    global_section = _build_global_section(outputs, assets_dir=assets_dir)
    if global_section is not None:
        sections.append(global_section)

    cohort_section = _build_cohort_section(outputs, assets_dir=assets_dir)
    if cohort_section is not None:
        sections.append(cohort_section)

    local_section = _build_local_section(
        outputs,
        selected_samples=selected_samples,
        assets_dir=assets_dir,
        show_original_per_explainer=bool(
            getattr(getattr(config, "reporting", None), "show_original_per_explainer", False)
        ),
        explicit_selection=explicit_samples is not None,
    )
    if local_section is not None:
        sections.append(local_section)

    robustness_section = _build_robustness_section(
        outputs,
        assets_dir=assets_dir,
        selected_samples=selected_samples,
        show_redundant_robustness_panels=bool(
            getattr(
                getattr(config, "reporting", None),
                "show_redundant_robustness_panels",
                False,
            )
        ),
    )
    if robustness_section is not None:
        sections.append(robustness_section)

    manifest = ReportManifest(
        kind="run",
        sections=tuple(sections),
        metadata={
            "experiment_name": getattr(config, "experiment_name", None),
            "selected_samples": [
                _selected_sample_manifest_entry(sample) for sample in selected_samples
            ],
        },
        filename=_manifest_filename(config),
    )
    return BuiltReport(report_dir=report_dir, sections=tuple(sections), manifest=manifest)


def build_merged_report(
    config: AppConfig,
    *,
    sweep_dir: Path,
    child_manifests: list[tuple[str, str | None, ReportManifest]],
    skipped_children: list[str],
) -> BuiltReport:
    report_dir = Path(sweep_dir) / "reports"
    assets_dir = report_dir / "_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the canonical single-run order for merged reports. Future/unknown
    # sections are appended after these headings in first-seen manifest order.
    sections_by_title: dict[str, list[ReportGroup]] = {
        "Metrics": [],
        "Global Explanations": [],
        "Cohort Explanations": [],
        "Local Explanations": [],
        "Robustness": [],
    }
    seen_metrics_rows: set[tuple[tuple[str, str], ...]] = set()

    for job_label, override_summary, manifest in child_manifests:
        prefix = job_label if not override_summary else f"{job_label} ({override_summary})"
        for section in manifest.sections:
            if section.title not in sections_by_title:
                sections_by_title[section.title] = []
            for group_index, group in enumerate(section.groups):
                if section.title == "Metrics" and group.table_rows:
                    metrics_key = tuple(group.table_rows)
                    if metrics_key in seen_metrics_rows:
                        continue
                    seen_metrics_rows.add(metrics_key)
                copied_images = tuple(
                    _copy_asset(
                        image_path,
                        assets_dir=assets_dir,
                        target_name=(
                            f"{job_label.lower().replace(' ', '_')}_"
                            f"{section.title.lower().replace(' ', '_')}_"
                            f"{group_index}_{idx}{image_path.suffix}"
                        ),
                    )
                    for idx, image_path in enumerate(group.images)
                )
                metadata = dict(group.metadata)
                metadata["child_job"] = job_label
                if override_summary:
                    metadata["override_summary"] = override_summary
                sections_by_title[section.title].append(
                    ReportGroup(
                        heading=f"{prefix} - {group.heading}",
                        images=copied_images,
                        table_rows=group.table_rows,
                        metadata=metadata,
                    )
                )

    ordered_sections = tuple(
        ReportSection.from_groups(
            title,
            groups,
            metadata={"section_role": title.lower().replace(" ", "_")},
        )
        for title, groups in sections_by_title.items()
        if groups
    )
    manifest = ReportManifest(
        kind="multirun",
        sections=ordered_sections,
        metadata={
            "experiment_name": getattr(config, "experiment_name", None),
            "children": [
                {"job_label": job_label, "override_summary": override_summary}
                for job_label, override_summary, _manifest in child_manifests
            ],
            "skipped_children": skipped_children,
        },
        filename=_manifest_filename(config),
    )
    return BuiltReport(report_dir=report_dir, sections=ordered_sections, manifest=manifest)


def _manifest_filename(config: AppConfig) -> str:
    reporting = getattr(config, "reporting", None)
    filename = str(getattr(reporting, "filename", "report.pdf"))
    target = str(getattr(reporting, "_target_", ""))
    if target in {"HTMLReporter", "raitap.reporting.HTMLReporter"}:
        return Path(filename).with_suffix(".html").name
    return filename


def _build_metrics_section(outputs: RunOutputs, *, assets_dir: Path) -> ReportSection | None:
    if outputs.metrics is None:
        return None

    source_group = outputs.metrics.to_report_group()
    staged_images = tuple(
        _copy_asset(
            image_path,
            assets_dir=assets_dir,
            target_name=f"metrics_{index}{image_path.suffix}",
        )
        for index, image_path in enumerate(source_group.images)
    )
    group = ReportGroup(
        heading=source_group.heading,
        images=staged_images,
        table_rows=source_group.table_rows,
        metadata={"role": "metrics"},
    )
    return ReportSection.from_groups("Metrics", [group], metadata={"section_role": "metrics"})


def _build_robustness_section(
    outputs: RunOutputs,
    *,
    assets_dir: Path,
    selected_samples: list[SelectedSample],
    show_redundant_robustness_panels: bool = False,
) -> ReportSection | None:
    """Build robustness report groups.

    Compact mode re-renders report-only variants through
    ``RobustnessResult.render_visualisation_for_report`` so duplicate empirical
    facets can be suppressed without changing persisted robustness artifacts or
    ``metadata.json``. This adds extra report-only renders only where compact
    layout kwargs are needed; legacy mode reuses pre-rendered artifacts.
    """
    if not outputs.robustness_results:
        return None

    visualisations_by_assessor: dict[str, list[RobustnessVisualisationResult]] = {}
    if show_redundant_robustness_panels:
        for visualisation in outputs.robustness_visualisations:
            assessor_name = visualisation.result.assessor_name or visualisation.result.run_dir.name
            visualisations_by_assessor.setdefault(assessor_name, []).append(visualisation)

    groups: list[ReportGroup] = []
    for index, result in enumerate(outputs.robustness_results):
        assessor_name = result.assessor_name or result.run_dir.name
        method_kind_value = result.method_kind.value
        if result.method_kind == MethodKind.EMPIRICAL_ATTACK:
            heading = f"Adversarial attack - {result.algorithm} ({assessor_name})"
        else:
            heading = f"Robustness certification - {result.algorithm} ({assessor_name})"

        budget = result.semantics.budget
        table_rows: list[tuple[str, str]] = [
            ("assessor", assessor_name),
            ("algorithm", result.algorithm),
            ("method_kind", method_kind_value),
            ("threat_model", result.semantics.threat_model.value),
            ("objective", result.semantics.objective.value),
            ("norm", budget.norm.value),
        ]
        if budget.epsilon is not None:
            table_rows.append(("epsilon", f"{budget.epsilon:g}"))
        for metric_name, metric_value in result.metrics.as_dict().items():
            table_rows.append((metric_name, f"{metric_value:.4f}"))
        table_rows.extend(_output_bounds_table_rows(result))

        robustness_sample_indices = _robustness_report_sample_indices(
            result,
            selected_samples=selected_samples,
        )
        staged_images = (
            _legacy_robustness_images(
                visualisations_by_assessor.get(assessor_name, []),
                assets_dir=assets_dir,
                result_index=index,
                assessor_name=assessor_name,
            )
            if show_redundant_robustness_panels
            else _compact_robustness_images(
                result,
                assets_dir=assets_dir,
                result_index=index,
                assessor_name=assessor_name,
                sample_indices=robustness_sample_indices,
            )
        )

        metadata: dict[str, object] = {
            "role": "robustness",
            "assessor_name": assessor_name,
            "algorithm": result.algorithm,
            "method_kind": method_kind_value,
        }
        if not show_redundant_robustness_panels:
            metadata["sample_indices"] = robustness_sample_indices

        groups.append(
            ReportGroup(
                heading=heading,
                images=tuple(staged_images),
                table_rows=tuple(table_rows),
                metadata=metadata,
            )
        )

    return ReportSection.from_groups(
        "Robustness",
        groups,
        metadata={"section_role": "robustness"},
    )


def _output_bounds_table_rows(result: Any) -> list[tuple[str, str]]:
    if result.method_kind != MethodKind.FORMAL_VERIFICATION:
        return []

    output_bounds = getattr(result, "output_bounds", None)
    if not isinstance(output_bounds, Mapping):
        return []

    lower = output_bounds.get("lower")
    upper = output_bounds.get("upper")
    if not isinstance(lower, torch.Tensor) or not isinstance(upper, torch.Tensor):
        return []
    if lower.ndim != 2 or upper.ndim != 2 or upper.shape != lower.shape:
        return []

    lower_values = lower.detach().cpu().to(torch.float32)
    upper_values = upper.detach().cpu().to(torch.float32)
    n_samples = int(lower_values.shape[0])
    lower_has_bounds = ~torch.isnan(lower_values).all(dim=1)
    upper_has_bounds = ~torch.isnan(upper_values).all(dim=1)
    per_sample_has_bounds = int((lower_has_bounds & upper_has_bounds).sum().item())
    rows = [("output_bounds_samples", f"{per_sample_has_bounds}/{n_samples}")]

    for logit_index in range(int(lower_values.shape[1])):
        lower_column = lower_values[:, logit_index]
        upper_column = upper_values[:, logit_index]
        if torch.isnan(lower_column).all() or torch.isnan(upper_column).all():
            continue
        rows.append(
            (
                f"logit_{logit_index}_lower_mean",
                f"{float(torch.nanmean(lower_column).item()):.4f}",
            )
        )
        rows.append(
            (
                f"logit_{logit_index}_upper_mean",
                f"{float(torch.nanmean(upper_column).item()):.4f}",
            )
        )

    return rows


def _legacy_robustness_images(
    visualisations: list[RobustnessVisualisationResult],
    *,
    assets_dir: Path,
    result_index: int,
    assessor_name: str,
) -> list[Path]:
    staged_images: list[Path] = []
    for visualisation in visualisations:
        staged_images.append(
            _copy_asset(
                visualisation.output_path,
                assets_dir=assets_dir,
                target_name=(
                    f"robustness_{result_index}_{_safe_name(assessor_name)}_"
                    f"{visualisation.visualiser_name}{visualisation.output_path.suffix}"
                ),
            )
        )
    return staged_images


def _compact_robustness_images(
    result: Any,
    *,
    assets_dir: Path,
    result_index: int,
    assessor_name: str,
    sample_indices: tuple[int, ...],
) -> list[Path]:
    configured_visualisers = list(result.visualisers)
    combined_visualiser_index = _combined_robustness_visualiser_index(configured_visualisers)
    owners = (
        {}
        if combined_visualiser_index is not None
        else _canonical_facet_owners(configured_visualisers)
    )
    staged_images: list[Path] = []
    sample_indices_for_render: tuple[int | None, ...] = (
        sample_indices if sample_indices else (None,)
    )
    for sample_index in sample_indices_for_render:
        for visualiser_index, configured in enumerate(configured_visualisers):
            if (
                combined_visualiser_index is not None
                and visualiser_index != combined_visualiser_index
            ):
                continue
            if combined_visualiser_index is not None:
                render_kwargs = {
                    "include_clean_input": True,
                    "include_perturbation_map": True,
                }
            else:
                render_kwargs = _render_kwargs_for_robustness_visualiser(
                    configured.visualiser,
                    owners=owners,
                    visualiser_index=visualiser_index,
                    omit_redundant=True,
                )
            visualisation = result.render_visualisation_for_report(
                visualiser_index,
                sample_index=sample_index,
                **render_kwargs,
            )
            if visualisation is None:
                continue
            sample_part = f"_sample_{sample_index}" if sample_index is not None else ""
            target = assets_dir / (
                f"robustness_{result_index}_{_safe_name(assessor_name)}"
                f"{sample_part}_{visualisation.visualiser_name}.png"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            _strip_report_figure_titles(visualisation.figure)
            try:
                visualisation.figure.savefig(target, bbox_inches="tight", dpi=150)
            finally:
                plt.close(visualisation.figure)
            staged_images.append(target)
    return staged_images


def _combined_robustness_visualiser_index(visualisers: Any) -> int | None:
    for index, configured in enumerate(visualisers):
        facets = set(_declared_robustness_facets(configured.visualiser))
        if {"clean_input", "perturbation_map"}.issubset(facets):
            return index
    return None


def _robustness_report_sample_indices(
    result: Any,
    *,
    selected_samples: list[SelectedSample],
) -> tuple[int, ...]:
    if result.method_kind != MethodKind.EMPIRICAL_ATTACK:
        return ()
    batch_size = int(result.clean_inputs.shape[0])
    return tuple(
        selected.summary.sample_index
        for selected in selected_samples
        if 0 <= selected.summary.sample_index < batch_size
    )


def _canonical_facet_owners(visualisers: Any) -> dict[str, int]:
    """Choose facet owners within one robustness result's visualiser list.

    When multiple visualisers embed the same facet, the owner is the first one
    that embeds *only* that facet. If none qualify, ownership prefers a
    candidate not already chosen for another facet so that visualisers embedding
    both facets don't all get told to omit both panels.
    """
    configured_visualisers = list(visualisers)
    owners: dict[str, int] = {}
    for facet in ("clean_input", "perturbation_map"):
        candidates: list[int] = []
        single_facet_owner: int | None = None
        for index, configured in enumerate(configured_visualisers):
            facets = _declared_robustness_facets(configured.visualiser)
            if facet not in facets:
                continue
            candidates.append(index)
            if len(facets) == 1 and single_facet_owner is None:
                single_facet_owner = index
        if not candidates:
            continue
        if single_facet_owner is not None:
            owners[facet] = single_facet_owner
            continue
        used = set(owners.values())
        for idx in candidates:
            if idx not in used:
                owners[facet] = idx
                break
        else:
            owners[facet] = candidates[0]
    return owners


def _declared_robustness_facets(visualiser: BaseRobustnessVisualiser) -> tuple[str, ...]:
    facets: list[str] = []
    cls = type(visualiser)
    if getattr(cls, "embeds_clean_input", False):
        facets.append("clean_input")
    if getattr(cls, "embeds_perturbation_map", False):
        facets.append("perturbation_map")
    return tuple(facets)


def _render_kwargs_for_robustness_visualiser(
    visualiser: BaseRobustnessVisualiser,
    *,
    owners: dict[str, int],
    visualiser_index: int,
    omit_redundant: bool,
) -> dict[str, object]:
    if not omit_redundant:
        return {}
    kwargs: dict[str, object] = {}
    cls = type(visualiser)
    if (
        getattr(cls, "embeds_clean_input", False)
        and owners.get("clean_input", visualiser_index) != visualiser_index
    ):
        kwargs["include_clean_input"] = False
    if (
        getattr(cls, "embeds_perturbation_map", False)
        and owners.get("perturbation_map", visualiser_index) != visualiser_index
    ):
        kwargs["include_perturbation_map"] = False
    if (
        kwargs.get("include_clean_input") is False
        and kwargs.get("include_perturbation_map") is False
    ):
        raise AssertionError(
            "Refusing to ask a robustness visualiser to omit both clean and perturbation panels."
        )
    return kwargs


def _build_global_section(outputs: RunOutputs, *, assets_dir: Path) -> ReportSection | None:
    groups = _native_scope_groups(
        outputs.visualisations,
        scope=ExplanationScope.GLOBAL,
        role="global",
        assets_dir=assets_dir,
    )
    if not groups:
        return None
    return ReportSection.from_groups(
        "Global Explanations",
        groups,
        metadata={"section_role": "global"},
    )


def _build_cohort_section(outputs: RunOutputs, *, assets_dir: Path) -> ReportSection | None:
    groups = _native_scope_groups(
        outputs.visualisations,
        scope=ExplanationScope.COHORT,
        role="cohort",
        assets_dir=assets_dir,
    )
    if not groups:
        return None
    return ReportSection.from_groups(
        "Cohort Explanations",
        groups,
        metadata={"section_role": "cohort"},
    )


def _native_scope_groups(
    visualisations: list[VisualisationResult],
    *,
    scope: ExplanationScope,
    role: str,
    assets_dir: Path,
) -> list[ReportGroup]:
    grouped: dict[str, list[Path]] = {}
    for visualisation in visualisations:
        if visualisation.scope != scope:
            continue
        explainer_name = (
            visualisation.explanation.explainer_name or visualisation.explanation.run_dir.name
        )
        staged = _copy_asset(
            visualisation.output_path,
            assets_dir=assets_dir,
            target_name=(
                f"{explainer_name}_{role}_"
                f"{visualisation.visualiser_name}{visualisation.output_path.suffix}"
            ),
        )
        grouped.setdefault(explainer_name, []).append(staged)

    return [
        ReportGroup(
            heading=f"Explainer: {explainer_name}",
            images=tuple(images),
            metadata={
                "role": role,
                "source": "native_visualiser",
                "explainer_name": explainer_name,
            },
        )
        for explainer_name, images in grouped.items()
    ]


def _build_local_section(
    outputs: RunOutputs,
    *,
    selected_samples: list[SelectedSample],
    assets_dir: Path,
    show_original_per_explainer: bool = False,
    explicit_selection: bool = False,
) -> ReportSection | None:
    if not outputs.explanations:
        return None

    if show_original_per_explainer or explicit_selection:
        return _build_legacy_local_section(
            outputs,
            selected_samples=selected_samples,
            assets_dir=assets_dir,
            explicit_selection=explicit_selection,
        )

    groups: list[ReportGroup] = []
    for selected in selected_samples:
        header_group = _build_sample_header_group(
            outputs,
            selected=selected,
            assets_dir=assets_dir,
            strip_titles=True,
        )
        omit_original = header_group is not None
        thumbnail_source_names = (
            (str(header_group.metadata["source_explainer_name"]),)
            if header_group is not None
            else ()
        )
        sample_groups: list[ReportGroup] = []

        for explanation in outputs.explanations:
            if not explanation.has_visualisations_for_scope(ExplanationScope.LOCAL):
                continue
            explainer_name = explanation.explainer_name or explanation.run_dir.name

            for visualiser_index, configured in enumerate(explanation.visualisers):
                if (
                    _scope_for_report_visualiser(explanation, configured.visualiser)
                    != ExplanationScope.LOCAL
                ):
                    continue

                render_kwargs = _render_kwargs_for_visualiser(
                    configured.visualiser,
                    omit_original=omit_original,
                )
                visualisation = explanation.render_visualisation_for_scope(
                    visualiser_index,
                    scope="local",
                    sample_index=selected.summary.sample_index,
                    **render_kwargs,
                )
                if visualisation is None:
                    continue

                images = _stage_rendered_visualisations(
                    [visualisation],
                    assets_dir=assets_dir,
                    file_stem_prefix=(
                        f"sample_{selected.summary.sample_index}_{_safe_name(explainer_name)}"
                    ),
                    strip_titles=True,
                )
                if not images:
                    continue

                visualiser_name = _visualiser_group_name(
                    configured.visualiser,
                    visualiser_index,
                )
                metadata: dict[str, object] = {
                    "role": "local_visualiser",
                    "bucket": selected.label,
                    "sample_index": selected.summary.sample_index,
                    "explainer_name": explainer_name,
                    "algorithm": explanation.algorithm,
                    "visualiser_name": visualiser_name,
                    "visualiser_index": visualiser_index,
                    "visualiser_class": type(configured.visualiser).__name__,
                }
                visualiser_title = getattr(configured.visualiser, "title", None)
                if visualiser_title:
                    metadata["visualiser_title"] = str(visualiser_title)
                if thumbnail_source_names:
                    metadata["thumbnail_source_explainer_names"] = thumbnail_source_names
                sample_groups.append(
                    ReportGroup(
                        heading=f"Explainer: {explainer_name} - Visualiser: {visualiser_name}",
                        images=images,
                        table_rows=_transparency_table_rows(
                            explanation,
                            selected_samples=[selected],
                            visualiser_index=visualiser_index,
                        ),
                        metadata=metadata,
                    )
                )

        if sample_groups:
            if header_group is not None:
                groups.append(header_group)
            groups.extend(sample_groups)

    if not groups:
        return None
    return ReportSection.from_groups(
        "Local Explanations",
        groups,
        metadata={"section_role": "local"},
    )


def _build_legacy_local_section(
    outputs: RunOutputs,
    *,
    selected_samples: list[SelectedSample],
    assets_dir: Path,
    explicit_selection: bool = False,
) -> ReportSection | None:
    groups: list[ReportGroup] = []
    overview_sample = (
        None if explicit_selection else selected_samples[0] if selected_samples else None
    )
    if overview_sample is not None:
        for explanation in outputs.explanations:
            rendered = explanation.render_visualisations_for_scope(
                scope="local",
                sample_index=overview_sample.summary.sample_index,
            )
            images = _stage_rendered_visualisations(
                rendered,
                assets_dir=assets_dir,
                file_stem_prefix=(
                    "overview_"
                    f"{_safe_name(explanation.explainer_name or explanation.run_dir.name)}_"
                    f"{overview_sample.summary.sample_index}"
                ),
            )
            if not images:
                continue
            groups.append(
                ReportGroup(
                    heading=(
                        "Overview - "
                        f"Explainer: {explanation.explainer_name or explanation.algorithm} - "
                        f"{_sample_label(overview_sample)}"
                    ),
                    images=images,
                    metadata={
                        "role": "local_overview",
                        "bucket": overview_sample.label,
                        "sample_index": overview_sample.summary.sample_index,
                        "selection_source": overview_sample.selection_source.value,
                        "explainer_name": explanation.explainer_name or explanation.run_dir.name,
                    },
                )
            )

    detail_samples = selected_samples[1:] if overview_sample is not None else selected_samples
    for selected in detail_samples:
        detail_images: list[Path] = []
        for explanation in outputs.explanations:
            rendered = explanation.render_visualisations_for_scope(
                scope="local",
                sample_index=selected.summary.sample_index,
            )
            detail_images.extend(
                _stage_rendered_visualisations(
                    rendered,
                    assets_dir=assets_dir,
                    file_stem_prefix=(
                        f"detail_{selected.label}_"
                        f"{_safe_name(explanation.explainer_name or explanation.run_dir.name)}_"
                        f"{selected.summary.sample_index}"
                    ),
                )
            )
        if not detail_images:
            continue
        groups.append(
            ReportGroup(
                heading=f"Detail - {_sample_label(selected)}",
                images=tuple(detail_images),
                metadata={
                    "role": "local_detail",
                    "bucket": selected.label,
                    "sample_index": selected.summary.sample_index,
                    "selection_source": selected.selection_source.value,
                    **_requested_sample_metadata(selected),
                },
            )
        )

    if not groups:
        return None
    return ReportSection.from_groups(
        "Local Explanations",
        groups,
        metadata={"section_role": "local"},
    )


def _build_sample_header_group(
    outputs: RunOutputs,
    *,
    selected: SelectedSample,
    assets_dir: Path,
    strip_titles: bool = False,
) -> ReportGroup | None:
    staged = _stage_sample_thumbnail(
        outputs,
        selected=selected,
        assets_dir=assets_dir,
        target_name=f"sample_{selected.summary.sample_index}_thumbnail_0.png",
        strip_titles=strip_titles,
    )
    if staged is None:
        return None
    return ReportGroup(
        heading=f"Sample - {_sample_label(selected)}",
        images=(staged.path,),
        table_rows=_sample_fact_rows(selected),
        metadata={
            "role": "sample_header",
            "bucket": selected.label,
            "sample_index": selected.summary.sample_index,
            "source_explainer_name": staged.source_explainer_name,
        },
    )


def _stage_sample_thumbnail(
    outputs: RunOutputs,
    *,
    selected: SelectedSample,
    assets_dir: Path,
    target_name: str,
    strip_titles: bool = False,
) -> _StagedThumbnail | None:
    visualiser = InputThumbnailVisualiser()
    sample_index = selected.summary.sample_index
    last_error: Exception | None = None

    for explanation in outputs.explanations:
        attributions = explanation.attributions[sample_index : sample_index + 1]
        inputs = (
            None
            if explanation.inputs is None
            else explanation.inputs[sample_index : sample_index + 1]
        )
        try:
            visualiser.validate_explanation(explanation, attributions, inputs)
        except ValueError as exc:
            last_error = exc
            continue

        sample_names = _sample_names_for_explanation(explanation, sample_index)
        show_sample_names = bool(explanation.kwargs.get("show_sample_names", False))
        context = VisualisationContext(
            algorithm=explanation.algorithm,
            sample_names=sample_names,
            show_sample_names=show_sample_names,
        )
        try:
            figure = visualiser.visualise(
                attributions,
                inputs=inputs,
                context=context,
                max_samples=1,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            raitap_log.exception(
                "Skipping sample thumbnail for sample %s: %s",
                sample_index,
                exc,
            )
            return None

        target = assets_dir / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        if strip_titles:
            _strip_report_figure_titles(figure)
        try:
            figure.savefig(target, bbox_inches="tight", dpi=150)
        finally:
            plt.close(figure)

        return _StagedThumbnail(
            path=target,
            source_explainer_name=explanation.explainer_name or explanation.run_dir.name,
        )

    if last_error is not None:
        raitap_log.warn(
            "Skipping sample thumbnail for sample %s: no compatible input visualiser (%s)",
            sample_index,
            last_error,
        )
    return None


def _transparency_table_rows(
    explanation: Any,
    *,
    selected_samples: Sequence[SelectedSample],
    visualiser_index: int,
) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    explainer_name = explanation.explainer_name or explanation.run_dir.name
    semantics = explanation.semantics
    input_spec = semantics.input_spec
    output_space = semantics.output_space

    rows.append(("explainer", str(explainer_name)))
    rows.append(("algorithm", str(explanation.algorithm)))

    method_families = _format_collection(
        _enum_value(method_family) for method_family in semantics.method_families
    )
    if method_families:
        rows.append(("method_families", method_families))

    targets = _format_selected_targets(explanation, selected_samples)
    if targets:
        rows.append(("targets", targets))

    if input_spec is not None:
        if input_spec.kind is not None:
            rows.append(("input_kind", _enum_value(input_spec.kind)))
        if input_spec.layout is not None:
            rows.append(("input_layout", _enum_value(input_spec.layout)))
        if input_spec.shape is not None:
            rows.append(("input_shape", _format_shape(input_spec.shape)))

    if output_space.space is not None:
        rows.append(("output_space", _enum_value(output_space.space)))
    if output_space.layout is not None:
        rows.append(("output_layout", _enum_value(output_space.layout)))
    if output_space.shape is not None:
        rows.append(("output_shape", _format_shape(output_space.shape)))
    if output_space.layer_path:
        rows.append(("layer_path", str(output_space.layer_path)))
    if output_space.requires_interpolation:
        rows.append(("requires_interpolation", "true"))

    for key, value in explanation.call_kwargs.items():
        formatted = _format_table_value(value)
        if formatted is not None:
            rows.append((f"call.{key}", formatted))

    configured = explanation.visualisers[visualiser_index]
    rows.extend(_visualiser_table_rows(configured.visualiser))
    for key, value in configured.call_kwargs.items():
        if key in _DISPLAY_ONLY_VISUALISER_KWARGS:
            continue
        formatted = _format_table_value(value)
        if formatted is not None:
            rows.append((f"visualiser_call.{key}", formatted))

    return tuple(rows)


def _visualiser_group_name(visualiser: BaseVisualiser, visualiser_index: int) -> str:
    title = getattr(visualiser, "title", None)
    if title:
        return str(title)
    return f"{type(visualiser).__name__}_{visualiser_index}"


def _scope_for_report_visualiser(
    explanation: Any,
    visualiser: BaseVisualiser,
) -> ExplanationScope:
    produces_scope = getattr(type(visualiser), "produces_scope", None)
    if isinstance(produces_scope, ExplanationScope):
        return produces_scope
    return explanation.semantics.scope


def _visualiser_table_rows(visualiser: BaseVisualiser) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [("visualiser_class", type(visualiser).__name__)]
    for attr, row_name in (
        ("title", "title"),
        ("method", "method"),
        ("sign", "sign"),
    ):
        if not hasattr(visualiser, attr):
            continue
        formatted = _format_table_value(getattr(visualiser, attr))
        if formatted is not None:
            rows.append((f"visualiser_{row_name}", formatted))

    return rows


def _format_selected_targets(
    explanation: Any,
    selected_samples: Sequence[SelectedSample],
) -> str | None:
    target_spec = getattr(explanation.semantics, "target", None)
    if target_spec is None:
        return None
    target_value = getattr(target_spec, "target", None)
    if target_value is None:
        return None

    parts: list[str] = []
    for selected in sorted(selected_samples, key=lambda item: item.summary.sample_index):
        sample_index = selected.summary.sample_index
        sample_target = _target_for_sample(target_value, sample_index)
        if sample_target is None:
            continue
        parts.append(f"{sample_index}: {_format_scalar(sample_target)}")
    return ", ".join(parts) if parts else None


def _target_for_sample(value: Any, sample_index: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        if 0 <= sample_index < int(value.shape[0]):
            item = value[sample_index]
            return item.item() if item.ndim == 0 else item.detach().cpu().tolist()
        return None
    if isinstance(value, Sequence) and not isinstance(value, str):
        if 0 <= sample_index < len(value):
            return value[sample_index]
        return None
    return value


def _format_table_value(value: Any) -> str | None:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _format_scalar(value.item())
        if value.numel() <= 8:
            return _format_table_value(value.detach().cpu().tolist())
        return None
    if isinstance(value, Enum):
        return _enum_value(value)
    if isinstance(value, bool):
        return _format_bool(value)
    if isinstance(value, str):
        return value
    if value is None:
        return None
    if isinstance(value, int | float):
        return _format_scalar(value)
    if isinstance(value, Mapping):
        formatted_items = []
        for key, item in value.items():
            formatted = _format_table_value(item)
            if formatted is not None:
                formatted_items.append(f"{key}: {formatted}")
        return ", ".join(formatted_items) if formatted_items else None
    if isinstance(value, Sequence) and not isinstance(value, str):
        if len(value) > 8:
            return None
        if all(isinstance(item, int) for item in value):
            return _format_shape(tuple(int(item) for item in value))
        return _format_collection(_format_scalar(item) for item in value)
    return str(value)


def _format_collection(values: Any) -> str:
    formatted = sorted(str(value) for value in values if value is not None)
    return ", ".join(formatted)


def _format_shape(shape: Sequence[int]) -> str:
    return " x ".join(str(part) for part in shape)


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_scalar(value: Any) -> str:
    if isinstance(value, Enum):
        return _enum_value(value)
    if isinstance(value, bool):
        return _format_bool(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _enum_value(value: Any) -> str:
    return str(value.value if isinstance(value, Enum) else value)


def _render_kwargs_for_visualiser(
    visualiser: BaseVisualiser,
    *,
    omit_original: bool,
) -> dict[str, object]:
    if not omit_original:
        return {}
    if not getattr(type(visualiser), "embeds_original_input", False):
        return {}
    if not visualiser.renders_attribution_only_when_original_hidden():
        return {}
    return {"include_original_input": False}


def _select_samples(outputs: RunOutputs) -> list[SelectedSample]:
    summaries = list(outputs.prediction_summaries)
    if summaries:
        selected: list[SelectedSample] = []
        seen: set[int] = set()

        for label, candidates, reverse in (
            ("wrong", [s for s in summaries if s.correct is False], True),
            ("insecure", summaries, False),
            ("high_confidence", [s for s in summaries if s.correct is not False], True),
        ):
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
        if selected:
            return selected

    total = _batch_size(outputs)
    if total <= 0:
        return []
    fallback_count = min(total, _MAX_LOCAL_DETAIL_SAMPLES)
    sample_ids = outputs.sample_ids or []
    selected = []
    for index in range(fallback_count):
        selected.append(
            SelectedSample(
                label="sample",
                summary=PredictionSummary(
                    sample_index=index,
                    predicted_class=-1,
                    confidence=0.0,
                    sample_id=sample_ids[index] if index < len(sample_ids) else None,
                ),
            )
        )
    return selected


def _explicit_selected_samples(
    resolved_samples: list[ResolvedReportSample],
    *,
    outputs: RunOutputs,
) -> list[SelectedSample]:
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
        for resolved in resolved_samples
    ]


def _batch_size(outputs: RunOutputs) -> int:
    if outputs.forward_output.ndim > 0:
        return int(outputs.forward_output.shape[0])
    return 0


def _copy_asset(source: Path, *, assets_dir: Path, target_name: str) -> Path:
    target_name_path = Path(target_name)
    if target_name_path.is_absolute() or len(target_name_path.parts) != 1:
        raise ValueError(f"Asset target names must be simple filenames, got {target_name!r}.")

    target = assets_dir / target_name_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _stage_rendered_visualisations(
    visualisations: list[VisualisationResult],
    *,
    assets_dir: Path,
    file_stem_prefix: str,
    strip_titles: bool = False,
) -> tuple[Path, ...]:
    staged: list[Path] = []
    for visualisation in visualisations:
        target = assets_dir / f"{file_stem_prefix}_{visualisation.visualiser_name}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if strip_titles:
            _strip_report_figure_titles(visualisation.figure)
        try:
            visualisation.figure.savefig(target, bbox_inches="tight", dpi=150)
        finally:
            plt.close(visualisation.figure)
        staged.append(target)
    return tuple(staged)


def _strip_report_figure_titles(figure: Any) -> None:
    if hasattr(figure, "suptitle"):
        figure.suptitle("")
    for ax in getattr(figure, "axes", []):
        ax.set_title("")


def _sample_names_for_explanation(explanation: object, sample_index: int) -> list[str]:
    kwargs = getattr(explanation, "kwargs", {})
    value = kwargs.get("sample_names") if isinstance(kwargs, dict) else None
    if value is None:
        return []
    try:
        names = [str(name) for name in value]
    except TypeError:
        return [str(value)]
    if sample_index < len(names):
        return [names[sample_index]]
    return []


def _sample_fact_rows(selected: SelectedSample) -> tuple[tuple[str, str], ...]:
    summary = selected.summary
    rows: list[tuple[str, str]] = [
        ("bucket", selected.label),
        ("sample_index", str(summary.sample_index)),
    ]
    if summary.sample_id:
        rows.append(("sample_id", summary.sample_id))
    if summary.predicted_class >= 0:
        rows.append(("predicted_class", str(summary.predicted_class)))
        rows.append(("confidence", f"{summary.confidence:.4f}"))
    if summary.target_class is not None:
        rows.append(("target_class", str(summary.target_class)))
    if summary.correct is not None:
        rows.append(("correct", str(summary.correct)))
    return tuple(rows)


def _sample_label(selected: SelectedSample) -> str:
    summary = selected.summary
    parts = [selected.label]
    if summary.sample_id:
        parts.append(summary.sample_id)
    else:
        parts.append(f"sample {summary.sample_index}")
    if summary.predicted_class >= 0:
        parts.append(f"pred={summary.predicted_class}")
        parts.append(f"conf={summary.confidence:.3f}")
    if summary.target_class is not None:
        parts.append(f"target={summary.target_class}")
    if summary.correct is not None:
        parts.append("correct" if summary.correct else "wrong")
    return " | ".join(parts)


def _selected_sample_manifest_entry(sample: SelectedSample) -> dict[str, object]:
    entry: dict[str, object] = {
        "label": sample.label,
        **asdict(sample.summary),
        "selection_source": sample.selection_source.value,
    }
    entry.update(_requested_sample_metadata(sample))
    return entry


def _requested_sample_metadata(sample: SelectedSample) -> dict[str, object]:
    if sample.requested_sample is None:
        return {}
    return {"requested_sample": sample.requested_sample}


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_") or "asset"
