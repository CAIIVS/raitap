from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from raitap.metrics.factory import MetricsEvaluation
from raitap.reporting.sections import ReportContext

if TYPE_CHECKING:
    from pathlib import Path


def _metrics_evaluation(run_dir: Path) -> MetricsEvaluation:
    fig = plt.figure()
    fig.savefig(run_dir / "overview.png")
    plt.close(fig)
    result = SimpleNamespace(metrics={"accuracy": 0.9123, "f1": 0.8})
    return MetricsEvaluation(
        result=cast("object", result),  # type: ignore[arg-type]
        run_dir=run_dir,
        computer=cast("object", None),  # type: ignore[arg-type]
        resolved_target="MulticlassClassificationMetrics",
    )


def test_metrics_evaluation_report_order() -> None:
    assert MetricsEvaluation.report_order == 10


def test_metrics_evaluation_report_sections(tmp_path: Path) -> None:
    evaluation = _metrics_evaluation(tmp_path)
    assets_dir = tmp_path / "_assets"
    assets_dir.mkdir()

    sections = evaluation.report_sections(ReportContext(assets_dir=assets_dir, selected_samples=()))

    assert [section.title for section in sections] == ["Metrics"]
    (group,) = sections[0].groups
    assert group.heading == "Performance Metrics"
    assert ("accuracy", "0.9123") in group.table_rows
    # Image staged into assets_dir under the metrics_<i> name.
    assert group.images and group.images[0].parent == assets_dir
    assert group.images[0].name == "metrics_0.png"
