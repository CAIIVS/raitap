"""Robustness phase result + report rendering.

``RobustnessPhaseResult`` is the robustness phase's contribution to a run: it is
``Trackable`` (logs its assessor results + visualisations) and ``Reportable``
(builds the "Robustness" report section). The phase class + work function that
produce it live in :mod:`raitap.robustness.phase`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import matplotlib.pyplot as plt

from raitap.reporting.sections import ReportGroup, ReportSection
from raitap.reporting.staging import _copy_asset, _safe_name, _strip_report_figure_titles
from raitap.robustness.contracts import (
    AssessmentKind,
    PerturbationBudget,
    PerturbationDistribution,
    ReportFigureScope,
)
from raitap.tracking.base_tracker import Trackable
from raitap.utils.lazy import lazy_import

if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from raitap.reporting.samples import SelectedSample
    from raitap.reporting.sections import ReportContext
    from raitap.robustness.results import RobustnessResult, RobustnessVisualisationResult
    from raitap.robustness.visualisers.base_visualiser import BaseRobustnessVisualiser
    from raitap.tracking.base_tracker import BaseTracker
else:
    torch = lazy_import("torch")


@dataclass
class RobustnessPhaseResult(Trackable):
    """Robustness phase output: the assessor results, each owning its visualisations.

    Each :class:`RobustnessResult` owns its ``.visualisations`` (issue #243) — the
    results are the single source of truth; there is no parallel phase-level
    visualisation list.
    """

    results: list[RobustnessResult] = field(default_factory=list)

    report_order: ClassVar[int] = 30

    def log(self, tracker: BaseTracker | None, **kwargs: Any) -> None:
        if tracker is None:
            return
        use_subdirs = len(self.results) > 1
        for result in self.results:
            result.log(tracker, use_subdirectory=use_subdirs)
            for visualisation in result.visualisations:
                visualisation.log(tracker, use_subdirectory=use_subdirs)

    def report_sections(self, ctx: ReportContext) -> tuple[ReportSection, ...]:
        section = _build_robustness_section(
            self,
            assets_dir=ctx.assets_dir,
            selected_samples=list(ctx.selected_samples),
            show_redundant_robustness_panels=ctx.show_redundant_robustness_panels,
        )
        return (section,) if section is not None else ()


def _build_robustness_section(
    outputs: RobustnessPhaseResult,
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
    layout kwargs are needed; verbose layout reuses pre-rendered artifacts.
    """
    if not outputs.results:
        return None

    groups: list[ReportGroup] = []
    for index, result in enumerate(outputs.results):
        assessor_name = result.name or result.run_dir.name
        assessment_kind_value = result.assessment_kind.value
        if result.assessment_kind == AssessmentKind.EMPIRICAL_ATTACK:
            heading = f"Adversarial attack - {result.algorithm} ({assessor_name})"
        elif result.assessment_kind == AssessmentKind.STATISTICAL_SAMPLING:
            heading = f"Average-case robustness - {result.algorithm} ({assessor_name})"
        else:
            heading = f"Robustness certification - {result.algorithm} ({assessor_name})"

        perturbation = result.semantics.perturbation
        table_rows: list[tuple[str, str]] = [
            ("assessor", assessor_name),
            ("algorithm", result.algorithm),
            ("assessment_kind", assessment_kind_value),
            ("case", result.semantics.case.value),
            ("threat_model", result.semantics.threat_model.value),
            ("objective", result.semantics.objective.value),
        ]
        if isinstance(perturbation, PerturbationBudget):
            table_rows.append(("norm", perturbation.norm.value))
            if perturbation.epsilon is not None:
                table_rows.append(("epsilon", f"{perturbation.epsilon:g}"))
        elif isinstance(perturbation, PerturbationDistribution):
            table_rows.append(("corruption_name", perturbation.corruption_name))
            table_rows.append(("severity", str(perturbation.severity)))
        for metric_name, metric_value in result.metrics.as_dict().items():
            table_rows.append((metric_name, f"{metric_value:.4f}"))
        table_rows.extend(_output_bounds_table_rows(result))

        robustness_sample_indices = _robustness_report_sample_indices(
            result,
            selected_samples=selected_samples,
        )
        staged = (
            _verbose_robustness_images(
                result.visualisations,
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
        staged_images = [path for path, _ in staged]
        # Scope is keyed by asset basename so the view model places each figure
        # (assessor-level vs per-sample) from data instead of parsing filenames.
        figure_scopes = {path.name: scope for path, scope in staged}

        metadata: dict[str, object] = {
            "role": "robustness",
            "assessor_name": assessor_name,
            "algorithm": result.algorithm,
            "assessment_kind": assessment_kind_value,
            "figure_scopes": figure_scopes,
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
    if result.assessment_kind != AssessmentKind.FORMAL_VERIFICATION:
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


def _figure_scope_for(result: Any, visualiser_name: str) -> str:
    """Resolve a staged figure's report scope from the configured visualiser.

    ``visualiser_name`` is ``{ClassName}_{index}``; the trailing index maps back to
    ``result.visualisers`` to read the declared ``report_figure_scope``. Falls back
    to per-sample when the index cannot be resolved.
    """
    default = ReportFigureScope.PER_SAMPLE.value
    _, _, suffix = visualiser_name.rpartition("_")
    try:
        index = int(suffix)
        configured = result.visualisers[index]
    except (ValueError, IndexError, AttributeError):
        return default
    return type(configured.visualiser).report_figure_scope.value


def _verbose_robustness_images(
    visualisations: list[RobustnessVisualisationResult],
    *,
    assets_dir: Path,
    result_index: int,
    assessor_name: str,
) -> list[tuple[Path, str]]:
    staged: list[tuple[Path, str]] = []
    for visualisation in visualisations:
        path = _copy_asset(
            visualisation.output_path,
            assets_dir=assets_dir,
            target_name=(
                f"robustness_{result_index}_{_safe_name(assessor_name)}_"
                f"{visualisation.visualiser_name}{visualisation.output_path.suffix}"
            ),
        )
        scope = _figure_scope_for(visualisation.result, visualisation.visualiser_name)
        staged.append((path, scope))
    return staged


def _compact_robustness_images(
    result: Any,
    *,
    assets_dir: Path,
    result_index: int,
    assessor_name: str,
    sample_indices: tuple[int, ...],
) -> list[tuple[Path, str]]:
    configured_visualisers = list(result.visualisers)
    combined_visualiser_index = _combined_robustness_visualiser_index(configured_visualisers)
    owners = (
        {}
        if combined_visualiser_index is not None
        else _canonical_facet_owners(configured_visualisers)
    )
    staged: list[tuple[Path, str]] = []
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
                if render_kwargs is None:
                    continue
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
            scope = type(configured.visualiser).report_figure_scope.value
            staged.append((target, scope))
    return staged


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
    if result.assessment_kind != AssessmentKind.EMPIRICAL_ATTACK:
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
) -> dict[str, object] | None:
    if not omit_redundant:
        return {}
    kwargs: dict[str, object] = {}
    declared_facets = set(_declared_robustness_facets(visualiser))
    disabled_facets: set[str] = set()
    cls = type(visualiser)
    if (
        getattr(cls, "embeds_clean_input", False)
        and owners.get("clean_input", visualiser_index) != visualiser_index
    ):
        kwargs["include_clean_input"] = False
        disabled_facets.add("clean_input")
    if (
        getattr(cls, "embeds_perturbation_map", False)
        and owners.get("perturbation_map", visualiser_index) != visualiser_index
    ):
        kwargs["include_perturbation_map"] = False
        disabled_facets.add("perturbation_map")
    if declared_facets and disabled_facets >= declared_facets:
        return None
    return kwargs
