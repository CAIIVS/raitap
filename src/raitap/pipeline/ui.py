"""Pipeline-specific UI: the assessment-summary banner printed before a run.

Generic console primitives (``Status``, ``colour``, panel helpers,
``_safe_attr``, ``_format_value``) live in :mod:`raitap.utils.console`. The
panel below is pipeline-specific because it reads ``config: AppConfig`` and
renders pipeline-specific rows (experiment, model, dataset, hardware,
explainers, robustness, metrics, output).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

from raitap.utils.console import (
    Status,
    _format_value,
    _safe_attr,
    colour,
    get_console,
)

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig
    from raitap.models import Model


def print_summary(config: AppConfig, model: Model) -> None:
    """Render the assessment-summary banner. Defensive against missing fields."""
    transparency = config.transparency or {}
    robustness = config.robustness or {}
    try:
        from raitap.metrics import metrics_run_enabled

        metrics_on = metrics_run_enabled(config)
    except Exception:
        metrics_on = False

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column(no_wrap=True, overflow="ellipsis")

    table.add_row("experiment", _format_value(_safe_attr(config, "experiment_name")))
    table.add_row("model", _format_value(_safe_attr(config, "model", "source")))
    table.add_row("dataset", _format_value(_safe_attr(config, "data", "name")))
    hardware = _safe_attr(model, "backend", "hardware_label")
    if hardware:
        hw_label = str(hardware)
        is_cpu = "cpu" in hw_label.lower()
        hw_status = Status.WARNING if is_cpu else Status.SUCCESS
        hw_style = colour(hw_status).base
        hw_symbol = hw_status.icon
        if is_cpu:
            requested = str(_safe_attr(config, "hardware") or "").lower()
            if requested == "cpu":
                hint_text = "CPU set in config"
                link_text = "View config docs"
                hint_link = "https://caiivs.github.io/raitap/using-raitap/configuration/global-config-options.html"
            else:
                hint_text = "GPU not usable, fell back to CPU"
                link_text = "View execution dependencies docs"
                hint_link = "https://caiivs.github.io/raitap/using-raitap/installation.html#execution-dependencies"
            hw_text = Text.assemble(
                (hw_symbol, hw_style),
                (hw_label, hw_style),
                (" · ", hw_style),
                (hint_text, hw_style),
                (" · ", hw_style),
                (link_text, hw_style + Style(underline=True, link=hint_link)),
            )
        else:
            hw_text = Text.assemble((hw_symbol, hw_style), (hw_label, hw_style))
    else:
        hw_text = Text("—", style="dim")
    table.add_row("hardware", hw_text)
    table.add_row("explainers", _format_value(list(transparency.keys())))
    table.add_row("robustness", _format_value(list(robustness.keys())))
    table.add_row("metrics", Text("on" if metrics_on else "off"))

    try:
        from pathlib import Path

        from raitap.configs import resolve_run_dir

        run_dir = str(resolve_run_dir(config))
        run_uri = Path(run_dir).resolve().as_uri()
        output_text = Text(run_dir, style=colour(Status.INFO).base + Style(link=run_uri))
    except Exception:
        # Banner must never crash the run — fall back to em-dash placeholder.
        output_text = Text("—", style="dim")
    table.add_row("output", output_text)

    info_style = colour(Status.INFO).base
    title = Text.assemble(
        ("RAITAP", info_style + Style(bold=True)),
        (" · Assessment summary", info_style),
    )
    panel = Panel(
        table,
        title=title,
        title_align="left",
        border_style=info_style,
        padding=(1, 2),
    )
    get_console().print()
    get_console().print(panel)
    get_console().print()
