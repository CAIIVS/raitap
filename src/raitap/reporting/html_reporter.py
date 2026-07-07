from __future__ import annotations

from importlib import resources
from typing import TYPE_CHECKING, Any

from raitap import __about__
from raitap.configs import resolve_run_dir
from raitap.reporting.registration import reporter

from .base_reporter import BaseReporter
from .filenames import report_output_filename
from .template_filters import as_dict, asr_band, bucket_class, fmt_num, fmt_pct, slug
from .view_model import build_view

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from .sections import ReportSection


@reporter(registry_name="html")
class HTMLReporter(BaseReporter):
    """Narrative HTML report generator using Jinja2."""

    def generate(
        self,
        sections: Sequence[ReportSection],
        *,
        report_dir: Path | None = None,
    ) -> Path:
        run_dir = (
            report_dir if report_dir is not None else resolve_run_dir(self.config, subdir="reports")
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        # HTMLReporter is only instantiated for reporting-enabled configs (see
        # ``factory.create_report``), so ``self.config.reporting`` is non-None here.
        assert self.config.reporting is not None
        html_path = run_dir / report_output_filename(
            self.config.reporting.filename,
            ".html",
        )
        metadata = {
            "experiment_name": self.config.experiment_name,
            "model_source": self.config.model.source,
            "data_name": self.config.data.name,
        }
        view = build_view(sections, metadata, version=__about__.__version__)
        env = _jinja_environment()
        css_text = _template_text("report.css")
        html = env.get_template("report.html.j2").render(view=view, css_text=css_text)
        html_path.write_text(html, encoding="utf-8")

        return html_path


def _jinja_environment() -> Any:
    from jinja2 import Environment, PackageLoader, select_autoescape

    env = Environment(
        loader=PackageLoader("raitap.reporting", "templates"),
        autoescape=select_autoescape(("html", "xml", "j2")),
    )
    env.filters.update(
        {
            "as_dict": as_dict,
            "asr_band": asr_band,
            "bucket_class": bucket_class,
            "fmt_num": fmt_num,
            "fmt_pct": fmt_pct,
            "slug": slug,
        }
    )
    return env


def _template_text(name: str) -> str:
    return resources.files("raitap.reporting.templates").joinpath(name).read_text(encoding="utf-8")
