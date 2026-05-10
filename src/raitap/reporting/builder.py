from __future__ import annotations

import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from raitap.configs import resolve_run_dir
from raitap.robustness.contracts import MethodKind
from raitap.run.outputs import PredictionSummary, RunOutputs
from raitap.transparency.contracts import ExplanationScope, VisualisationContext
from raitap.transparency.visualisers import BaseVisualiser, InputThumbnailVisualiser

from .manifest import ReportManifest
from .sections import ReportGroup, ReportSection

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig
    from raitap.robustness.results import RobustnessVisualisationResult
    from raitap.robustness.visualisers.base_visualiser import BaseRobustnessVisualiser
    from raitap.transparency.results import VisualisationResult

logger = logging.getLogger(__name__)

# Caps both the selected sample pool and, after reserving the first item for the
# overview, the number of detail groups shown in the local section.
_MAX_LOCAL_DETAIL_SAMPLES = 3


@dataclass(frozen=True)
class SelectedSample:
    label: str
    summary: PredictionSummary


@dataclass(frozen=True)
class BuiltReport:
    report_dir: Path
    sections: tuple[ReportSection, ...]
    manifest: ReportManifest


def build_report(config: AppConfig, outputs: RunOutputs) -> BuiltReport:
    report_dir = resolve_run_dir(config, subdir="reports")
    assets_dir = report_dir / "_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    selected_samples = _select_samples(outputs)
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
    )
    if local_section is not None:
        sections.append(local_section)

    robustness_section = _build_robustness_section(
        outputs,
        assets_dir=assets_dir,
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
                {"label": sample.label, **asdict(sample.summary)} for sample in selected_samples
            ],
        },
        filename=getattr(getattr(config, "reporting", None), "filename", "report.pdf"),
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
        filename=getattr(getattr(config, "reporting", None), "filename", "report.pdf"),
    )
    return BuiltReport(report_dir=report_dir, sections=ordered_sections, manifest=manifest)


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
            )
        )

        groups.append(
            ReportGroup(
                heading=heading,
                images=tuple(staged_images),
                table_rows=tuple(table_rows),
                metadata={
                    "role": "robustness",
                    "assessor_name": assessor_name,
                    "algorithm": result.algorithm,
                    "method_kind": method_kind_value,
                },
            )
        )

    return ReportSection.from_groups(
        "Robustness",
        groups,
        metadata={"section_role": "robustness"},
    )


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
) -> list[Path]:
    owners = _canonical_facet_owners(result.visualisers)
    staged_images: list[Path] = []
    for visualiser_index, configured in enumerate(result.visualisers):
        render_kwargs = _render_kwargs_for_robustness_visualiser(
            configured.visualiser,
            owners=owners,
            visualiser_index=visualiser_index,
            omit_redundant=True,
        )
        visualisation = result.render_visualisation_for_report(
            visualiser_index,
            **render_kwargs,
        )
        if visualisation is None:
            continue
        target = assets_dir / (
            f"robustness_{result_index}_{_safe_name(assessor_name)}_"
            f"{visualisation.visualiser_name}.png"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            visualisation.figure.savefig(target, bbox_inches="tight", dpi=150)
        finally:
            plt.close(visualisation.figure)
        staged_images.append(target)
    return staged_images


def _canonical_facet_owners(visualisers: Any) -> dict[str, int]:
    """Choose facet owners within one robustness result's visualiser list."""
    configured_visualisers = list(visualisers)
    owners: dict[str, int] = {}
    for facet in ("clean_input", "perturbation_map"):
        first_embedder: int | None = None
        for index, configured in enumerate(configured_visualisers):
            facets = _declared_robustness_facets(configured.visualiser)
            if facet not in facets:
                continue
            if first_embedder is None:
                first_embedder = index
            if len(facets) == 1:
                owners[facet] = index
                break
        else:
            if first_embedder is not None:
                owners[facet] = first_embedder
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
) -> ReportSection | None:
    if not outputs.explanations:
        return None

    if show_original_per_explainer:
        return _build_legacy_local_section(
            outputs,
            selected_samples=selected_samples,
            assets_dir=assets_dir,
        )

    groups: list[ReportGroup] = []
    for selected in selected_samples:
        header_group = _build_sample_header_group(
            outputs,
            selected=selected,
            assets_dir=assets_dir,
        )
        omit_original = header_group is not None

        sample_groups: list[ReportGroup] = []
        for explanation in outputs.explanations:
            explainer_name = explanation.explainer_name or explanation.run_dir.name
            rendered = []
            for visualiser_index, configured in enumerate(explanation.visualisers):
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
                if visualisation is not None:
                    rendered.append(visualisation)

            images = _stage_rendered_visualisations(
                rendered,
                assets_dir=assets_dir,
                file_stem_prefix=(
                    f"sample_{selected.summary.sample_index}_"
                    f"{_safe_name(explainer_name)}"
                ),
            )
            if not images:
                continue
            sample_groups.append(
                ReportGroup(
                    heading=f"Explainer: {explainer_name} - {_sample_label(selected)}",
                    images=images,
                    metadata={
                        "role": "local_attribution",
                        "bucket": selected.label,
                        "sample_index": selected.summary.sample_index,
                        "explainer_name": explainer_name,
                    },
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
) -> ReportSection | None:
    groups: list[ReportGroup] = []
    overview_sample = selected_samples[0] if selected_samples else None
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
) -> ReportGroup | None:
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
            logger.warning(
                "Skipping sample thumbnail for sample %s: %s",
                sample_index,
                exc,
                exc_info=True,
            )
            return None

        target = assets_dir / f"sample_{sample_index}_thumbnail_0.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            figure.savefig(target, bbox_inches="tight", dpi=150)
        finally:
            plt.close(figure)

        return ReportGroup(
            heading=f"Sample - {_sample_label(selected)}",
            images=(target,),
            table_rows=_sample_fact_rows(selected),
            metadata={
                "role": "sample_header",
                "bucket": selected.label,
                "sample_index": sample_index,
                "source_explainer_name": explanation.explainer_name or explanation.run_dir.name,
            },
        )

    if last_error is not None:
        logger.warning(
            "Skipping sample thumbnail for sample %s: no compatible input visualiser (%s)",
            sample_index,
            last_error,
        )
    return None


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
) -> tuple[Path, ...]:
    staged: list[Path] = []
    for visualisation in visualisations:
        target = assets_dir / f"{file_stem_prefix}_{visualisation.visualiser_name}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            visualisation.figure.savefig(target, bbox_inches="tight", dpi=150)
        finally:
            plt.close(visualisation.figure)
        staged.append(target)
    return tuple(staged)


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


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_") or "asset"
