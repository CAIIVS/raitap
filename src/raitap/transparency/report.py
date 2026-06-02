"""Transparency phase result + report rendering.

``TransparencyPhaseResult`` is the transparency phase's contribution to a run:
it is ``Trackable`` (logs its explanations + visualisations) and ``Reportable``
(builds the Global / Aggregated / Local explanation sections). The
section-building helpers were relocated verbatim from ``reporting/builder.py`` —
the only change is that they read explanations/visualisations off the
``TransparencyPhaseResult`` passed as ``outputs`` instead of the old flat
``RunOutputs`` fields.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

import matplotlib.pyplot as plt

from raitap import raitap_log
from raitap.pipeline.phases.assess_transparency import assess_transparency
from raitap.pipeline.phases.base import AssessmentPhase
from raitap.reporting.samples import _requested_sample_metadata
from raitap.reporting.sections import ReportGroup, ReportSection
from raitap.reporting.staging import (
    _copy_asset,
    _safe_name,
    _stage_rendered_visualisations,
    _strip_report_figure_titles,
)
from raitap.tracking.base_tracker import Trackable
from raitap.transparency.contracts import DetectionBox, ExplanationScope, VisualisationContext
from raitap.transparency.visualisers import BaseVisualiser, InputThumbnailVisualiser
from raitap.utils.lazy import lazy_import

if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from raitap.configs.schema import AppConfig
    from raitap.pipeline.outputs import PhaseResult
    from raitap.pipeline.phases.base import PhaseContext
    from raitap.reporting.samples import SelectedSample
    from raitap.reporting.sections import ReportContext
    from raitap.tracking.base_tracker import BaseTracker
    from raitap.transparency.results import ExplanationResult, VisualisationResult
else:
    torch = lazy_import("torch")


class TransparencyPhase(AssessmentPhase):
    name = "transparency"

    def is_configured(self, config: AppConfig) -> bool:
        return bool(getattr(config, "transparency", None))

    def run(self, ctx: PhaseContext) -> PhaseResult | None:
        explanations, visualisations = assess_transparency(
            ctx.config,
            ctx.model,
            ctx.data,
            ctx.forward_output,
            input_metadata=ctx.input_metadata,
            resolved_preprocessing=ctx.resolved_preprocessing,
        )
        return TransparencyPhaseResult(explanations=explanations, visualisations=visualisations)

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


@dataclass(frozen=True)
class _StagedThumbnail:
    path: Path
    source_explainer_name: str


@dataclass
class TransparencyPhaseResult(Trackable):
    """Transparency phase output: explanations + their flattened visualisations."""

    explanations: list[ExplanationResult] = field(default_factory=list)
    visualisations: list[VisualisationResult] = field(default_factory=list)

    report_order: ClassVar[int] = 20

    def log(self, tracker: BaseTracker | None, **kwargs: Any) -> None:
        if tracker is None:
            return
        use_subdirs = len(self.explanations) > 1
        for explanation in self.explanations:
            explanation.log(tracker, use_subdirectory=use_subdirs)
        for visualisation in self.visualisations:
            visualisation.log(tracker, use_subdirectory=use_subdirs)

    def report_sections(self, ctx: ReportContext) -> tuple[ReportSection, ...]:
        sections: list[ReportSection] = []
        global_section = _build_global_section(self, assets_dir=ctx.assets_dir)
        if global_section is not None:
            sections.append(global_section)
        aggregated_section = _build_aggregated_section(self, assets_dir=ctx.assets_dir)
        if aggregated_section is not None:
            sections.append(aggregated_section)
        local_section = _build_local_section(
            self,
            selected_samples=list(ctx.selected_samples),
            assets_dir=ctx.assets_dir,
            show_original_per_explainer=ctx.show_original_per_explainer,
            explicit_selection=ctx.explicit_selection,
        )
        if local_section is not None:
            sections.append(local_section)
        return tuple(sections)


def _build_global_section(
    outputs: TransparencyPhaseResult, *, assets_dir: Path
) -> ReportSection | None:
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


def _build_aggregated_section(
    outputs: TransparencyPhaseResult, *, assets_dir: Path
) -> ReportSection | None:
    groups = _native_scope_groups(
        outputs.visualisations,
        scope=ExplanationScope.AGGREGATED,
        role="aggregated",
        assets_dir=assets_dir,
    )
    if not groups:
        return None
    return ReportSection.from_groups(
        "Aggregated Explanations",
        groups,
        metadata={"section_role": "aggregated"},
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


def _detection_box_heading(box: DetectionBox) -> str:
    """One-line ``Box`` heading clause: predicted label + optional ground-truth match.

    Shows the ``gt:`` clause only when GT was evaluated for the box's sample —
    a matched box reads ``pred: X 0.99 | gt: Y (IoU ..)``; a no-match box reads
    ``| gt: no match`` (neutral, not "false positive" — GT may be incomplete);
    with no GT, the legacy ``label, score=..`` form is unchanged.
    """
    label = box.label_name or f"class {box.label_index}"
    if not box.ground_truth_evaluated:
        return f"{label}, score={box.score:.2f}"
    if box.true_label_index is None:
        ground_truth_clause = "no match"
    else:
        ground_truth_name = box.true_label_name or f"class {box.true_label_index}"
        # ``true_match_iou`` is normally set alongside the index by the matcher;
        # guard the format in case a caller populates only the label.
        ground_truth_clause = (
            f"{ground_truth_name} (IoU {box.true_match_iou:.2f})"
            if box.true_match_iou is not None
            else ground_truth_name
        )
    return f"pred: {label} {box.score:.2f} | gt: {ground_truth_clause}"


def _build_local_section(
    outputs: TransparencyPhaseResult,
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
            box_suffix_for_baseline = (
                f"_box_{explanation.detection_box.display_index}_{explanation.detection_box.raw_index}"
                if explanation.detection_box is not None
                else ""
            )
            baseline_image = _stage_baseline_image(
                explanation,
                assets_dir=assets_dir,
                stem=(
                    f"sample_{selected.summary.sample_index}_"
                    f"{_safe_name(explainer_name)}{box_suffix_for_baseline}"
                ),
            )
            baseline_emitted = False

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

                # Detection K-loop emits one ExplanationResult per box, all
                # sharing the same (sample_index, explainer, visualiser). Add
                # the box's display / raw index to the asset filename + group
                # heading + metadata so per-box panels don't collide.
                detection_box = explanation.detection_box
                box_suffix = (
                    f"_box_{detection_box.display_index}_{detection_box.raw_index}"
                    if detection_box is not None
                    else ""
                )
                images = _stage_rendered_visualisations(
                    [visualisation],
                    assets_dir=assets_dir,
                    file_stem_prefix=(
                        f"sample_{selected.summary.sample_index}_{_safe_name(explainer_name)}"
                        f"{box_suffix}"
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
                heading_box_part = ""
                if detection_box is not None:
                    heading_box_part = (
                        f" - Box {detection_box.display_index}"
                        f" ({_detection_box_heading(detection_box)})"
                    )
                    metadata["detection_box"] = {
                        "display_index": detection_box.display_index,
                        "raw_index": detection_box.raw_index,
                        "score": detection_box.score,
                        "label_index": detection_box.label_index,
                        "label_name": detection_box.label_name,
                        "xyxy": list(detection_box.xyxy),
                        "ground_truth_evaluated": detection_box.ground_truth_evaluated,
                        "true_label_index": detection_box.true_label_index,
                        "true_label_name": detection_box.true_label_name,
                        "true_match_iou": detection_box.true_match_iou,
                    }
                visualiser_title = getattr(configured.visualiser, "title", None)
                if visualiser_title:
                    metadata["visualiser_title"] = str(visualiser_title)
                if thumbnail_source_names:
                    metadata["thumbnail_source_explainer_names"] = thumbnail_source_names
                group_images = images
                if baseline_image is not None:
                    # One field drives BOTH the "View baseline" link (every card)
                    # and the anchored image in the reference card, so they cannot
                    # drift into a dead anchor (issue #210 review #6). Stored as str
                    # so group metadata stays JSON-serialisable for the manifest.
                    metadata["baseline_image"] = str(baseline_image)
                    if not baseline_emitted:
                        # Add the file to group.images once per explanation so the
                        # PDF reporter (which renders all group images) shows it once.
                        group_images = (*images, baseline_image)
                        baseline_emitted = True
                sample_groups.append(
                    ReportGroup(
                        heading=(
                            f"Explainer: {explainer_name}"
                            f" - Visualiser: {visualiser_name}"
                            f"{heading_box_part}"
                        ),
                        images=group_images,
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
    outputs: TransparencyPhaseResult,
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
    outputs: TransparencyPhaseResult,
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
    metadata: dict[str, object] = {
        "role": "sample_header",
        "bucket": selected.label,
        "sample_index": selected.summary.sample_index,
        "selection_source": selected.selection_source.value,
        "source_explainer_name": staged.source_explainer_name,
    }
    metadata.update(_requested_sample_metadata(selected))
    return ReportGroup(
        heading=f"Sample - {_sample_label(selected)}",
        images=(staged.path,),
        table_rows=_sample_fact_rows(selected),
        metadata=metadata,
    )


def _stage_sample_thumbnail(
    outputs: TransparencyPhaseResult,
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
        if explanation.original_sample_index is not None:
            # Detection K-loop: each ExplanationResult is single-sample
            # (attributions shape ``(1, ...)``) tied to one sample. Use the
            # whole tensor when it matches the requested sample; skip when
            # it represents a different sample (slicing would produce empty).
            if explanation.original_sample_index != sample_index:
                continue
            attributions = explanation.attributions
            inputs = explanation.inputs
        else:
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
            last_error = exc
            continue

        # Detection samples: overlay every sibling box onto the thumbnail so
        # the pinned section doubles as a navigation overview (all K boxes
        # at a glance — per-box attribution figures follow in the local
        # section).
        if explanation.original_sample_index is not None:
            _overlay_detection_boxes(figure, outputs=outputs, sample_index=sample_index)

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


def _overlay_detection_boxes(
    figure: Any, *, outputs: TransparencyPhaseResult, sample_index: int
) -> None:
    """Draw every detection box for a sample onto the thumbnail figure."""
    from matplotlib import patches as mpatches

    boxes = [
        explanation.detection_box
        for explanation in outputs.explanations
        if explanation.detection_box is not None
        and explanation.original_sample_index == sample_index
    ]
    if not boxes:
        return
    axes = figure.axes
    if not axes:
        return
    ax = axes[0]
    # On the image: only a compact ``#index`` tag per box (anchored inside its
    # top-left corner) so labels never collide when boxes are close. The full
    # per-box detail goes in a legend below the image, keyed by the same index.
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy
        ax.add_patch(
            mpatches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
        )
        ax.text(
            x1 + 2,
            y1 + 2,
            f"#{box.display_index}",
            color="black",
            fontsize=8,
            fontweight="bold",
            va="top",
            ha="left",
            bbox={"facecolor": "lime", "alpha": 0.9, "pad": 1, "edgecolor": "none"},
        )
    # Legend below the image (``bbox_inches="tight"`` at save time keeps it in
    # frame). One line per box; close boxes stay legible because their detail
    # lives here, not stacked on the image.
    legend = "\n".join(_overlay_legend_line(box) for box in boxes)
    ax.text(
        0.0,
        -0.02,
        legend,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        family="monospace",
        color="#222222",
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "boxstyle": "round,pad=0.4"},
        clip_on=False,
    )


def _overlay_legend_line(box: DetectionBox) -> str:
    """One legend row for a detection box: ``#i name (score) [| gt: ...]``."""
    label = box.label_name or f"class {box.label_index}"
    base = f"#{box.display_index} {label} ({box.score:.2f})"
    if not box.ground_truth_evaluated:
        return base
    if box.true_label_index is None:
        return f"{base} | gt: no match"
    ground_truth_name = box.true_label_name or f"class {box.true_label_index}"
    if box.true_match_iou is None:  # label without a match IoU — show name alone
        return f"{base} | gt: {ground_truth_name}"
    return f"{base} | gt: {ground_truth_name} (IoU {box.true_match_iou:.2f})"


# Human-facing labels for the baseline ``mode`` token. The raw token is kept in
# ``metadata.json``; the report shows the readable phrasing (issue #210).
_BASELINE_MODE_LABELS = {
    "configured": "configured dataset",
    "user_tensor": "user-provided tensor",
    "zero": "all-zeros (method default)",
    "input_batch": "input batch (method default)",
}


def _baseline_mode_label(mode: object) -> str:
    token = str(mode)
    return _BASELINE_MODE_LABELS.get(token, token)


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

    baseline = getattr(explanation, "baseline", None)
    if baseline is not None:
        rows.append(("baseline.mode", _baseline_mode_label(baseline.mode)))
        if baseline.source is not None:
            rows.append(("baseline.source", str(baseline.source)))
        if baseline.n_samples is not None:
            rows.append(("baseline.n_samples", str(baseline.n_samples)))
        rows.append(("baseline.shape", _format_shape(baseline.shape)))

    for key, value in explanation.call_kwargs.items():
        if baseline is not None and key == baseline.kwarg_name:
            continue
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


def _stage_baseline_image(
    explanation: Any,
    *,
    assets_dir: Path,
    stem: str,
) -> Path | None:
    """Copy an explanation's baseline PNG into ``assets_dir``; return staged path.

    Returns ``None`` when the explanation has no baseline image or the source
    file is missing (backward-compatible artefacts).
    """
    baseline = getattr(explanation, "baseline", None)
    if baseline is None or baseline.image_path is None:
        return None
    source = explanation.run_dir / baseline.image_path
    if not source.exists():
        return None
    assets_dir.mkdir(parents=True, exist_ok=True)
    target = assets_dir / f"baseline_{stem}{source.suffix}"
    shutil.copy2(source, target)
    return target
