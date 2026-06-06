"""Assemble a run's report from its phase results.

``build_report`` is generic dispatch: it computes the highlighted samples once
(phase-agnostic, from the run's predictions) and then asks each
:class:`~raitap.pipeline.outputs.PhaseResult` for its sections via the
``Reportable`` protocol, ordered by ``report_order``. The per-phase rendering
lives in each module (``transparency/report.py``, ``robustness/report.py``,
``metrics/factory.py``), so adding a module needs no edits here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from raitap.configs import resolve_run_dir
from raitap.reproducibility import reproducibility_caveat, stochastic_methods

from .filenames import report_output_filename
from .manifest import ReportManifest
from .sample_selection import resolve_report_sample_selection
from .samples import (
    EdgecaseSelectorStrategy,
    SampleSelectionStrategy,
    SelectedSample,
    UserSelectorStrategy,
    _requested_sample_metadata,
    report_batch_size,
)
from .sections import ReportContext, ReportGroup, ReportSection
from .staging import _copy_asset

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig
    from raitap.pipeline.outputs import RunOutputs


@dataclass(frozen=True)
class BuiltReport:
    report_dir: Path
    sections: tuple[ReportSection, ...]
    manifest: ReportManifest


def build_report(config: AppConfig, outputs: RunOutputs) -> BuiltReport:
    report_dir = resolve_run_dir(config, subdir="reports")
    assets_dir = report_dir / "_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    reporting_cfg = config.reporting
    # ``reporting`` may arrive as a struct-mode DictConfig that omits the
    # optional ``sample_selection`` key; read defensively so the field's
    # default applies instead of raising ``ConfigAttributeError`` (sibling of
    # #240; matches the orchestrator's getattr guard for the same key).
    configured_selection = (
        None if reporting_cfg is None else getattr(reporting_cfg, "sample_selection", None)
    )
    explicit_samples = resolve_report_sample_selection(
        configured_selection,
        sample_ids=outputs.sample_ids,
        batch_size=report_batch_size(outputs),
    )
    sample_strategy: SampleSelectionStrategy = (
        UserSelectorStrategy(explicit_samples)
        if explicit_samples is not None
        else EdgecaseSelectorStrategy()
    )
    selected_samples = sample_strategy.select(outputs)

    report_ctx = ReportContext(
        assets_dir=assets_dir,
        selected_samples=tuple(selected_samples),
        show_original_per_explainer=bool(
            getattr(reporting_cfg, "show_original_per_explainer", False)
        ),
        show_redundant_robustness_panels=bool(
            getattr(reporting_cfg, "show_redundant_robustness_panels", False)
        ),
        explicit_selection=(
            explicit_samples is not None
            and _reporting_target(config) in {"PDFReporter", "raitap.reporting.PDFReporter"}
        ),
    )

    sections: list[ReportSection] = []
    banner = _reproducibility_banner(outputs)
    if banner is not None:
        sections.append(banner)
    ordered_results = sorted(outputs.phase_results.values(), key=lambda result: result.report_order)
    for phase_result in ordered_results:
        sections.extend(phase_result.report_sections(report_ctx))

    manifest = ReportManifest(
        kind="run",
        sections=tuple(sections),
        metadata={
            "experiment_name": getattr(config, "experiment_name", None),
            "model_source": getattr(getattr(config, "model", None), "source", None),
            "data_name": getattr(getattr(config, "data", None), "name", None),
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
        "Aggregated Explanations": [],
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
            "model_source": getattr(getattr(config, "model", None), "source", None),
            "data_name": getattr(getattr(config, "data", None), "name", None),
            "children": [
                {"job_label": job_label, "override_summary": override_summary}
                for job_label, override_summary, _manifest in child_manifests
            ],
            "skipped_children": skipped_children,
        },
        filename=_manifest_filename(config),
    )
    return BuiltReport(report_dir=report_dir, sections=ordered_sections, manifest=manifest)


def _reproducibility_banner(outputs: RunOutputs) -> ReportSection | None:
    """Run-level reproducibility caveat, or ``None`` when the run is fully deterministic.

    Emitted as a ``ReportSection`` (not view metadata) because ``sections`` is the
    only channel both the HTML and PDF reporters consume. Prepended so it renders
    first. See issue #251.
    """
    methods = stochastic_methods(outputs)
    if not methods:
        return None
    return ReportSection.from_groups(
        "Reproducibility",
        [ReportGroup(heading=reproducibility_caveat(methods))],
        metadata={"section_role": "reproducibility"},
    )


def _manifest_filename(config: AppConfig) -> str:
    target = _reporting_target(config)
    filename = str(getattr(getattr(config, "reporting", None), "filename", "report"))
    if target in {"HTMLReporter", "raitap.reporting.HTMLReporter"}:
        return report_output_filename(filename, ".html")
    return report_output_filename(filename, ".pdf")


def _reporting_target(config: AppConfig) -> str:
    reporting = getattr(config, "reporting", None)
    return str(getattr(reporting, "_target_", ""))


def _selected_sample_manifest_entry(sample: SelectedSample) -> dict[str, object]:
    entry: dict[str, object] = {
        "label": sample.label,
        **asdict(sample.summary),
        "selection_source": sample.selection_source.value,
    }
    entry.update(_requested_sample_metadata(sample))
    return entry
