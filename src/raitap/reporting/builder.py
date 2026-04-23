from __future__ import annotations

import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from raitap.configs import resolve_run_dir
from raitap.run.outputs import PredictionSummary, RunOutputs
from raitap.transparency.visualisers import TabularBarChartVisualiser

from .manifest import ReportManifest
from .sections import ReportGroup, ReportSection

if TYPE_CHECKING:
    import torch

    from raitap.configs.schema import AppConfig
    from raitap.transparency.results import ExplanationResult, VisualisationResult

logger = logging.getLogger(__name__)

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

    local_section = _build_local_section(
        outputs,
        selected_samples=selected_samples,
        assets_dir=assets_dir,
    )
    if local_section is not None:
        sections.append(local_section)

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

    sections_by_title: dict[str, list[ReportGroup]] = {
        "Metrics": [],
        "Global Explanations": [],
        "Local Explanations": [],
    }

    for job_label, override_summary, manifest in child_manifests:
        prefix = job_label if not override_summary else f"{job_label} ({override_summary})"
        for section in manifest.sections:
            if section.title not in sections_by_title:
                sections_by_title[section.title] = []
            for group_index, group in enumerate(section.groups):
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


def _build_global_section(outputs: RunOutputs, *, assets_dir: Path) -> ReportSection | None:
    groups: list[ReportGroup] = []
    native_groups = _native_global_groups(outputs.visualisations, assets_dir=assets_dir)
    groups.extend(native_groups)

    if _batch_size(outputs) > 1:
        for explanation in outputs.explanations:
            aggregate_asset = _build_aggregate_asset(explanation, assets_dir=assets_dir)
            if aggregate_asset is None:
                continue
            groups.append(
                ReportGroup(
                    heading=f"Explainer: {explanation.explainer_name or explanation.algorithm}",
                    images=(aggregate_asset,),
                    metadata={
                        "role": "global",
                        "source": "raitap_aggregate",
                        "explainer_name": explanation.explainer_name or explanation.run_dir.name,
                    },
                )
            )

    if not groups:
        return None
    return ReportSection.from_groups(
        "Global Explanations",
        groups,
        metadata={"section_role": "global"},
    )


def _native_global_groups(
    visualisations: list[VisualisationResult], *, assets_dir: Path
) -> list[ReportGroup]:
    grouped: dict[str, list[Path]] = {}
    for visualisation in visualisations:
        if visualisation.report_scope != "global":
            continue
        explainer_name = (
            visualisation.explanation.explainer_name or visualisation.explanation.run_dir.name
        )
        staged = _copy_asset(
            visualisation.output_path,
            assets_dir=assets_dir,
            target_name=f"{explainer_name}_global_{visualisation.visualiser_name}{visualisation.output_path.suffix}",
        )
        grouped.setdefault(explainer_name, []).append(staged)

    return [
        ReportGroup(
            heading=f"Explainer: {explainer_name}",
            images=tuple(images),
            metadata={
                "role": "global",
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
) -> ReportSection | None:
    if not outputs.explanations:
        return None

    groups: list[ReportGroup] = []
    overview_sample = selected_samples[0] if selected_samples else None
    if overview_sample is not None:
        for explanation in outputs.explanations:
            images = explanation.save_visualisations_for_report(
                assets_dir,
                scope="local",
                file_stem_prefix=(
                    "overview_"
                    f"{_safe_name(explanation.explainer_name or explanation.run_dir.name)}_"
                    f"{overview_sample.summary.sample_index}"
                ),
                sample_index=overview_sample.summary.sample_index,
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
            detail_images.extend(
                explanation.save_visualisations_for_report(
                    assets_dir,
                    scope="local",
                    file_stem_prefix=(
                        f"detail_{selected.label}_"
                        f"{_safe_name(explanation.explainer_name or explanation.run_dir.name)}_"
                        f"{selected.summary.sample_index}"
                    ),
                    sample_index=selected.summary.sample_index,
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


def _build_aggregate_asset(explanation: ExplanationResult, *, assets_dir: Path) -> Path | None:
    if not explanation.has_visualisations_for_scope("local"):
        return None

    attrs = explanation.attributions
    batch_size = int(attrs.shape[0]) if attrs.ndim > 0 else 0
    if batch_size <= 1:
        return None

    explainer_name = explanation.explainer_name or explanation.run_dir.name
    target_name = f"{_safe_name(explainer_name)}_aggregate.png"
    output_path = assets_dir / target_name

    if attrs.ndim >= 4:
        _save_image_aggregate(
            attrs,
            explanation.inputs,
            output_path=output_path,
            title=f"{explainer_name} Mean Absolute Attribution",
        )
        return output_path
    if attrs.ndim == 2:
        _save_tabular_visualiser_aggregate(
            attrs,
            output_path=output_path,
            title=f"{explainer_name} Mean Absolute Attribution",
        )
        return output_path
    return None


def _save_image_aggregate(
    attributions: torch.Tensor,
    inputs: torch.Tensor,
    *,
    output_path: Path,
    title: str,
) -> None:
    mean_attr = attributions.abs().mean(dim=0)
    heatmap = mean_attr.sum(dim=0).cpu().numpy() if mean_attr.ndim == 3 else mean_attr.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    if inputs.ndim == 4 and inputs.shape[0] > 1:
        mean_input = inputs.mean(dim=0).cpu()
        if mean_input.ndim == 3:
            image = mean_input.permute(1, 2, 0).numpy()
            image_min = float(image.min())
            image_max = float(image.max())
            if image_max > image_min:
                image = (image - image_min) / (image_max - image_min)
            axes[0].imshow(image)
        else:
            axes[0].imshow(mean_input.numpy(), cmap="gray")
        axes[0].set_title("Mean Input")
        axes[0].axis("off")
    else:
        axes[0].imshow(heatmap, cmap="viridis")
        axes[0].set_title("Mean Absolute Attribution")
        axes[0].axis("off")

    image = axes[1].imshow(heatmap, cmap="magma")
    axes[1].set_title("Attribution Heatmap")
    axes[1].axis("off")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _save_tabular_visualiser_aggregate(
    attributions: torch.Tensor, *, output_path: Path, title: str
) -> None:
    visualiser = TabularBarChartVisualiser()
    figure = visualiser.visualise(attributions, title=title)
    try:
        figure.savefig(output_path, bbox_inches="tight", dpi=150)
    finally:
        plt.close(figure)


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
    target = assets_dir / target_name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


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
