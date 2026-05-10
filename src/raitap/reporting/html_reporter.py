from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from raitap import __about__
from raitap.configs import resolve_run_dir

from .base_reporter import BaseReporter
from .template_filters import as_dict, asr_band, bucket_class, fmt_num, fmt_pct, slug
from .view_model import build_view

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .sections import ReportSection


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

        html_path = run_dir / _html_filename(
            getattr(self.config.reporting, "filename", "report.pdf")
        )
        css_path = run_dir / "report.css"

        metadata = {
            "experiment_name": getattr(self.config, "experiment_name", None),
        }
        view = build_view(sections, metadata, version=__about__.__version__)
        env = _jinja_environment()
        css_text = _template_text("report.css")
        # Sibling copy keeps the HTML standalone for browser viewing.
        css_path.write_text(css_text, encoding="utf-8")
        html = env.get_template("report.html.j2").render(view=view)
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


def _html_filename(filename: str) -> str:
    path = Path(filename)
    return path.with_suffix(".html").name


def _template_text(name: str) -> str:
    return resources.files("raitap.reporting.templates").joinpath(name).read_text(encoding="utf-8")
