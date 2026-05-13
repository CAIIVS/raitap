"""Render the raitap-deps preview as a Rich panel matching the run banner."""

from __future__ import annotations

import sys

from rich.console import Group
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

from raitap.utils.colour import Status, colour
from raitap.utils.console import get_console


def _python_label(pinned: str | None) -> str:
    host = f"{sys.version_info.major}.{sys.version_info.minor}"
    if pinned is None:
        return f"{host} (host default)"
    if pinned == host:
        return f"{pinned} (pinned, matches host)"
    return f"{pinned} (pinned, host is {host})"


def print_deps_frame(
    *,
    hardware: str,
    hardware_origin: str,
    python_version: str | None,
    extras: list[str],
    pretty_command: str,
    action: str,
    note_blocks: list[Text] | None = None,
) -> None:
    """Print a Rich panel summarising the dep-bootstrap decision.

    ``action`` is a short verb shown in the header chip. ``note_blocks``
    (when provided) are rendered below the table separated by blank lines —
    each block is a pre-styled :class:`rich.text.Text`, so callers control
    which spans are warning-coloured (prose) vs white (copy-paste commands).
    """
    info = colour(Status.INFO).base

    table = Table.grid(padding=(0, 2), pad_edge=False)
    table.add_column(style="dim", justify="right")
    table.add_column(no_wrap=False, overflow="fold")
    table.add_row("hardware", Text(f"{hardware} ({hardware_origin})", style=info))
    table.add_row("python", Text(_python_label(python_version), style=info))
    table.add_row("extras", Text(", ".join(extras) if extras else "(none)", style=info))
    table.add_row("command", Text(pretty_command, style=Style(color="white")))

    title = Text.assemble(
        ("RAITAP", info + Style(bold=True)),
        (" · Deps · ", info),
        (action, info),
    )

    body: object = table
    if note_blocks:
        renderables: list[object] = [table, Text("")]
        for i, block in enumerate(note_blocks):
            if i:
                renderables.append(Text(""))
            renderables.append(block)
        body = Group(*renderables)

    panel = Panel(body, title=title, title_align="left", border_style=info, padding=(1, 2))
    console = get_console()
    console.print()
    console.print(panel)
    console.print()


def print_deps_error_frame(*, label: str, message: str, details: list[str] | None = None) -> None:
    """Render a deps-failure panel matching the ``Status.ERROR`` styling."""
    err = colour(Status.ERROR).base
    body = Text()
    body.append(f"{Status.ERROR.icon}", style=err)
    body.append(message, style=err + Style(bold=True))
    for detail in details or []:
        body.append("\n  • ", style=err)
        body.append(detail, style=err)

    title = Text.assemble(
        ("RAITAP", err + Style(bold=True)),
        (" · Deps · ", err),
        (label, err),
    )
    panel = Panel(body, title=title, title_align="left", border_style=err, padding=(1, 2))
    console = get_console()
    console.print()
    console.print(panel)
    console.print()
