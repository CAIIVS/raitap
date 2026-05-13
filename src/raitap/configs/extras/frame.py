"""Render the raitap-deps preview as a Rich panel matching the run banner."""

from __future__ import annotations

from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

from raitap.utils.colour import Status, colour
from raitap.utils.console import get_console


def print_deps_frame(
    *,
    hardware: str,
    hardware_origin: str,
    python_version: str | None,
    extras: list[str],
    pretty_command: str,
    action: str,
) -> None:
    """Print a Rich panel summarising the dep-bootstrap decision.

    ``action`` is a short verb shown in the header chip — typically
    ``"sync"`` or ``"dry-run"``.
    """
    info = colour(Status.INFO).base

    table = Table.grid(padding=(0, 2), pad_edge=False)
    table.add_column(style="dim", justify="right")
    table.add_column(no_wrap=False, overflow="fold")
    table.add_row("hardware", Text(f"{hardware} ({hardware_origin})", style=info))
    table.add_row("python", Text(python_version or "host default", style=info))
    table.add_row("extras", Text(", ".join(extras) if extras else "(none)", style=info))
    table.add_row("command", Text(pretty_command, style=Style(color="white")))

    title = Text.assemble(
        ("RAITAP", info + Style(bold=True)),
        (" · Deps · ", info),
        (action, info),
    )
    panel = Panel(table, title=title, title_align="left", border_style=info, padding=(1, 2))
    console = get_console()
    console.print()
    console.print(panel)
    console.print()
