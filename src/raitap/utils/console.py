"""Cross-platform terminal output helpers built on :mod:`rich`.

Rich auto-detects terminal capabilities (Windows console VT support, pipes,
``NO_COLOR``/``FORCE_COLOR``/``TERM=dumb``) so the same code path produces
ANSI-colored boxes on a real terminal and clean ASCII when redirected.
"""

from __future__ import annotations

import contextlib
import logging
import re
import sys
import warnings
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TextColumn,
)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.traceback import install as install_rich_traceback

from raitap.utils.diagnostics import (
    Diagnostic,
    docs_url,
    is_dev_install,
    resolve_diagnostic_from_frames,
)
from raitap.utils.warnings import _pop_diagnostic, _push_diagnostic, _take_diagnostic_override

_THEME = Theme(
    {
        "log.time": "cyan",
        "log.message": "default",
        "logging.level.info": "cyan",
        "logging.level.warning": "bright_yellow",
        "logging.level.error": "red",
    }
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from raitap.configs.schema import AppConfig
    from raitap.models import Model

_console: Console | None = None
_stderr_console: Console | None = None

# Source-prefix from `warnings.formatwarning`: "<path>:<line>: <Category>: msg".
_WARNING_PREFIX_RE = re.compile(
    r"^(?P<src>.+?:\d+):\s*(?P<cat>\w+Warning):\s*(?P<msg>.*)$",
    re.DOTALL,
)

# Detect Windows-drive or POSIX absolute paths inside arbitrary log messages.
_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|(?<![\w/])/)[^\s]+")
_PATH_TRIM = '.,;:)]}"'

# "Header: rest" pattern for non-warnings.warn WARNING records.
# Example match: ``logger.warning("Robustness: no labels…")``.
_LOGGER_HEADER_RE = re.compile(r"^(?P<head>[A-Z][\w\- ]{1,40}):\s+(?P<msg>.+)$", re.DOTALL)


def _src_to_uri(src: str) -> str:
    """Build a ``file://`` URI from a ``path:line`` string for OSC 8 hyperlinks."""
    from pathlib import Path

    path_only = src.rsplit(":", 1)[0] if ":" in src else src
    try:
        return Path(path_only).as_uri()
    except (ValueError, OSError):
        normalized = path_only.replace("\\", "/")
        return f"file:///{normalized.lstrip('/')}"


def _linkify_message(message: str) -> Text:
    """Wrap any path-like substrings in a cyan OSC 8 hyperlink."""
    from pathlib import Path

    rendered = Text()
    last = 0
    for match in _PATH_RE.finditer(message):
        rendered.append(message[last : match.start()])
        raw = match.group(0)
        trimmed = raw.rstrip(_PATH_TRIM)
        trailing = raw[len(trimmed) :]
        try:
            uri = Path(trimmed).as_uri()
        except (ValueError, OSError):
            # Non-absolute or invalid path — fall back to manual file:// URI.
            normalized = trimmed.replace("\\", "/")
            uri = f"file:///{normalized.lstrip('/')}"
        rendered.append(trimmed, style=f"cyan link {uri}")
        if trailing:
            rendered.append(trailing)
        last = match.end()
    rendered.append(message[last:])
    return rendered


def _reconfigure_utf8(stream: Any) -> None:
    """Best-effort: switch stdout/stderr to UTF-8 with ``errors='replace'`` so
    box-drawing/dot glyphs never crash on legacy Windows code pages (cp1252)."""
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    # Stream may be redirected/closed; safely no-op in that case.
    with contextlib.suppress(ValueError, OSError):
        reconfigure(encoding="utf-8", errors="replace")


def get_console() -> Console:
    global _console
    if _console is None:
        _reconfigure_utf8(sys.stdout)
        _console = Console(soft_wrap=False, highlight=False, theme=_THEME)
    return _console


def get_stderr_console() -> Console:
    global _stderr_console
    if _stderr_console is None:
        _reconfigure_utf8(sys.stderr)
        _stderr_console = Console(stderr=True, soft_wrap=False, highlight=False, theme=_THEME)
    return _stderr_console


_LEVEL_ICONS: dict[int, tuple[str, str]] = {
    logging.DEBUG: ("·", "dim"),
    logging.INFO: ("▸", "cyan"),
    logging.WARNING: ("▲", "bright_yellow"),
    logging.ERROR: ("✗", "red"),
    logging.CRITICAL: ("✗", "bold red"),
}


class RaitapRichHandler(RichHandler):
    """RichHandler with icon level-prefix and Panel rendering for multi-line warnings/errors."""

    _last_was_panel: bool = False

    def get_level_text(self, record: logging.LogRecord) -> Text:
        icon, style = _LEVEL_ICONS.get(record.levelno, ("·", "white"))
        return Text(icon, style=style)

    def render_message(self, record: logging.LogRecord, message: str) -> Any:
        return _linkify_message(message)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            super().emit(record)
            return

        # Frame every WARNING+ record (warnings.warn, logger.warning, errors).
        if record.levelno >= logging.WARNING:
            self._emit_panel(record, message)
            return
        super().emit(record)
        self._last_was_panel = False

    def _emit_panel(self, record: logging.LogRecord, message: str) -> None:
        if record.levelno >= logging.ERROR:
            border, main_style, sub_style, icon, label = "red", "red", "yellow", "✗", "Error"
        else:
            border, main_style, sub_style, icon, label = (
                "yellow",
                "yellow",
                "bright_yellow",
                "▲",
                "Warning",
            )

        body = message
        header_parts = [f"[{main_style}]{icon} {label}[/]"]
        # Only treat the "<path>:<line>: <Category>: msg" shape as a warnings.warn
        # payload when it actually came from logging.captureWarnings (logger name
        # ``py.warnings``). Otherwise an unrelated WARNING that happens to match
        # the pattern would desync the per-thread origin queue and mislabel the
        # next real warning.
        is_py_warning = record.name == "py.warnings"
        warn_match = _WARNING_PREFIX_RE.match(message.strip()) if is_py_warning else None
        if warn_match:
            cat = warn_match.group("cat")
            src = warn_match.group("src")
            diagnostic = _pop_diagnostic()
            self._render_warning_header(
                header_parts,
                src=src,
                category=cat,
                diagnostic=diagnostic,
                main_style=main_style,
                sub_style=sub_style,
            )
            body = warn_match.group("msg").strip()
        else:
            head_match = _LOGGER_HEADER_RE.match(message.strip())
            if head_match:
                head = head_match.group("head")
                header_parts.append(f"[{sub_style}]· {head}[/]")
                # logger.warning("Subsystem: …") records have no Diagnostic
                # stashed (no frame walk happened), so synthesise one from the
                # header text to drive the same View-docs affordance.
                synth = Diagnostic(
                    subsystem=head.lower(),
                    file="",
                    line=0,
                    third_party_lib=None,
                )
                if not is_dev_install():
                    url = docs_url(synth)
                    if url is not None:
                        header_parts.append(
                            f"[{sub_style}]· [/][{sub_style} underline link={url}]View docs[/]"
                        )
                stripped = head_match.group("msg").strip()
                body = stripped[:1].upper() + stripped[1:] if stripped else stripped

        # Build a non-wrapping ``Text`` with overflow="ellipsis" so a narrow
        # terminal trims chips with ``…`` instead of hard-cutting mid-word.
        title_text = Text.from_markup(" ".join(header_parts))
        title_text.overflow = "ellipsis"
        title_text.no_wrap = True
        panel = Panel(
            _linkify_message(body),
            title=title_text,
            title_align="left",
            border_style=border,
            padding=(0, 1),
        )
        try:
            if not self._last_was_panel:
                self.console.print()
            self.console.print(panel)
            self.console.print()
            self._last_was_panel = True
        except Exception:
            # Never let logging crash the run — fall back to plain RichHandler.
            super().emit(record)

    def _render_warning_header(
        self,
        header_parts: list[str],
        *,
        src: str,
        category: str,
        diagnostic: Diagnostic | None,
        main_style: str,
        sub_style: str,
    ) -> None:
        """Append subsystem / path / docs-link chips to the warning panel header.

        Layout depends on the audience:

        - **Dev install** (cloned repo): ``· <Subsystem> · <path:line>``,
          ``path`` clickable. Falls back to category if no subsystem matched.
        - **Installed wheel, raitap-emitted**: ``· <Subsystem>`` only,
          subsystem text linked to the docs page.
        - **Installed wheel, third-party**: ``· <Subsystem> · via <lib>``,
          subsystem linked to the frameworks-and-libraries doc page.
        - **Installed wheel, unclassified**: no subheader at all.
        """
        sub = diagnostic.subsystem if diagnostic else None
        third_party = diagnostic.third_party_lib if diagnostic else None
        scope = sub.capitalize() if sub else category

        if is_dev_install():
            header_parts.append(f"[{sub_style}]· {scope}[/]")
            header_parts.append(
                f"[{main_style}]· [/][{main_style} link={_src_to_uri(src)}]{src}[/]"
            )
            if third_party is not None:
                header_parts.append(f"[{sub_style}]· via {third_party.capitalize()}[/]")
            return

        # Installed: hide raw paths the user can't act on. Surface docs links instead.
        if sub is None:
            return
        header_parts.append(f"[{sub_style}]· {scope}[/]")
        if third_party is not None:
            header_parts.append(f"[{sub_style}]· via {third_party.capitalize()}[/]")
        url = docs_url(diagnostic) if diagnostic is not None else None
        if url is not None:
            # Explicit ``View docs`` chip — clearer affordance than relying on
            # OSC 8 styling on the subsystem text alone (Windows Terminal /
            # other terminals don't visually mark hyperlinks by default).
            # Link wraps only the label so the leading separator stays plain.
            header_parts.append(f"[{sub_style}]· [/][{sub_style} underline link={url}]View docs[/]")


def setup_logging(level: int = logging.INFO) -> None:
    """Install a single :class:`RaitapRichHandler` on the root logger.

    Also routes ``warnings.warn`` through the logging system so multi-line
    warnings (e.g. from captum, shap, torchattacks) get framed Panel output
    instead of raw stderr noise. ``force=True`` clears handlers Hydra/MLflow
    may have attached.
    """
    handler = RaitapRichHandler(
        console=get_console(),
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        markup=False,
        log_time_format="[%X]",
    )
    _reconfigure_utf8(sys.stdout)
    _reconfigure_utf8(sys.stderr)
    logging.basicConfig(level=level, format="%(message)s", handlers=[handler], force=True)
    logging.captureWarnings(True)
    # Compact one-line format for warnings.warn (path:line: Category: msg) — our
    # handler then unpacks it back into a Panel with subtitle = source.
    warnings.formatwarning = _format_warning_compact  # type: ignore[assignment]
    install_rich_traceback(console=get_stderr_console(), show_locals=False, width=None)


def _format_warning_compact(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    line: str | None = None,
) -> str:
    """Resolve the warn origin to the first raitap subsystem frame and stash it
    for :class:`RaitapRichHandler`.

    The returned canonical string ``path:line: Category: msg`` is what the
    standard ``logging.captureWarnings`` pipeline forwards to the handler. The
    structured :class:`~raitap.utils.diagnostics.Diagnostic` (subsystem +
    third-party detection) is stashed on a thread-local queue so the handler
    can render an audience-appropriate header without re-walking frames at
    emit time (frames are unwound by then).
    """
    diagnostic = _take_diagnostic_override() or resolve_diagnostic_from_frames(filename, lineno)
    _push_diagnostic(diagnostic)
    return f"{diagnostic.file}:{diagnostic.line}: {category.__name__}: {message}"


def _format_value(value: Any, *, dot: bool = False, dot_style: str = "green") -> Text:
    """Render an arbitrary config value, gracefully handling ``None``/empty.

    ``dot=True`` prepends a colored ``●`` glyph (used for status fields).
    """
    if value is None or value == "" or value == [] or value == {}:
        return Text("—", style="dim")
    if isinstance(value, bool):
        if value:
            if dot:
                return Text.assemble(("● ", dot_style), ("on", "green"))
            return Text("on", style="green")
        return Text("off", style="dim")
    if isinstance(value, list | tuple):
        return Text(", ".join(str(item) for item in value))
    text = Text(str(value))
    if dot:
        return Text.assemble(("● ", dot_style), text)
    return text


def _safe_attr(obj: Any, *path: str, default: Any = None) -> Any:
    cur = obj
    for name in path:
        if cur is None:
            return default
        cur = getattr(cur, name, None)
    return default if cur is None else cur


def print_summary_panel(config: AppConfig, model: Model) -> None:
    """Render the startup banner. Defensive against missing/None fields."""
    transparency = getattr(config, "transparency", None) or {}
    robustness = getattr(config, "robustness", None) or {}
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
        # CPU runs are slow enough to be a footgun for transparency/robustness;
        # surface that with the same yellow used for warnings.
        hw_label = str(hardware)
        hw_color = "yellow" if "cpu" in hw_label.lower() else "green"
        hw_text = Text.assemble(("● ", hw_color), (hw_label, hw_color))
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
        output_text = Text(run_dir, style=f"cyan link {run_uri}")
    except Exception:
        # Banner must never crash the run — fall back to em-dash placeholder.
        output_text = Text("—", style="dim")
    table.add_row("output", output_text)

    panel = Panel(
        table,
        title="[bold cyan]RAITAP[/] [cyan]· assessment[/]",
        title_align="left",
        border_style="cyan",
        padding=(1, 2),
    )
    get_console().print()
    get_console().print(panel)
    get_console().print()


def print_complete_panel(duration: str) -> None:
    body = Text.assemble(
        ("✓  ", "bold green"),
        ("Assessment complete", "bold"),
        ("    duration ", "dim"),
        (duration, "white"),
    )
    panel = Panel(
        body,
        border_style="bold green",
        padding=(0, 2),
    )
    get_console().print()
    get_console().print(panel)
    get_console().print()


def print_failure_panel(exc: BaseException, duration: str) -> None:
    body = Text.assemble(
        ("✗  ", "bold red"),
        ("Assessment failed", "bold"),
        ("    after ", "dim"),
        (duration, "white"),
        ("\n", ""),
        (f"{type(exc).__name__}: {exc}", "red"),
    )
    panel = Panel(
        body,
        border_style="bold red",
        padding=(0, 2),
    )
    get_stderr_console().print()
    get_stderr_console().print(panel)
    get_stderr_console().print()


class _PercentColumn(ProgressColumn):
    def render(self, task: Any) -> Text:
        style = "green" if task.finished else "cyan"
        pct = 0.0 if task.percentage is None else task.percentage
        return Text(f"{pct:>3.0f}%", style=style)


class _ElapsedColumn(ProgressColumn):
    def render(self, task: Any) -> Text:
        style = "green" if task.finished else "cyan"
        elapsed = task.finished_time if task.finished else task.elapsed
        text = "-:--:--" if elapsed is None else str(timedelta(seconds=int(elapsed)))
        return Text(text, style=style)


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[cyan]{task.description}"),
        BarColumn(complete_style="cyan", finished_style="green"),
        _PercentColumn(),
        _ElapsedColumn(),
        console=get_console(),
        transient=False,
    )


def iter_with_progress(
    iterable: Iterable[Any],
    *,
    total: int | None,
    desc: str,
) -> Iterator[Any]:
    """Iterate ``iterable`` while rendering a raitap-themed progress bar.

    Drop-in replacement for ``tqdm`` in for-loop usage. Returns a generator
    that owns the :class:`~rich.progress.Progress` lifecycle.
    """
    progress = _make_progress()
    with progress:
        task = progress.add_task(desc, total=total)
        for item in iterable:
            yield item
            progress.advance(task)
