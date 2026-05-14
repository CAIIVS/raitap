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
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TextColumn,
)
from rich.style import Style
from rich.text import Text
from rich.traceback import install as install_rich_traceback

from raitap.utils.colour import THEME, Status, colour
from raitap.utils.diagnostics import (
    Diagnostic,
    docs_url,
    is_dev_install,
    resolve_diagnostic_from_frames,
    resolve_diagnostic_from_path,
    resolve_diagnostic_from_traceback,
    subsystem_from_str,
)
from raitap.utils.errors import RaitapError
from raitap.utils.log import _pop_diagnostic, _push_diagnostic, _take_diagnostic_override
from raitap.utils.status_frame import StatusFrame, chip

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


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
_BACKTICK_RE = re.compile(r"`([^`\n]+)`")

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
    """Style backtick-quoted code spans magenta and wrap path-like substrings in
    a cyan (Status.INFO) OSC 8 hyperlink."""
    from pathlib import Path

    rendered = Text()
    last = 0
    for match in _PATH_RE.finditer(message):
        rendered.append_text(_stylize_inline_code(message[last : match.start()]))
        raw = match.group(0)
        trimmed = raw.rstrip(_PATH_TRIM)
        trailing = raw[len(trimmed) :]
        try:
            uri = Path(trimmed).as_uri()
        except (ValueError, OSError):
            # Non-absolute or invalid path — fall back to manual file:// URI.
            normalized = trimmed.replace("\\", "/")
            uri = f"file:///{normalized.lstrip('/')}"
        rendered.append(trimmed, style=colour(Status.INFO).base + Style(link=uri))
        if trailing:
            rendered.append(trailing)
        last = match.end()
    rendered.append_text(_stylize_inline_code(message[last:]))
    return rendered


def _stylize_inline_code(text: str) -> Text:
    """Render backtick-quoted runs in magenta; strip the surrounding backticks."""
    out = Text()
    last = 0
    for match in _BACKTICK_RE.finditer(text):
        out.append(text[last : match.start()])
        out.append(match.group(1), style="magenta")
        last = match.end()
    out.append(text[last:])
    return out


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
        _console = Console(soft_wrap=False, highlight=False, theme=THEME)
    return _console


def get_stderr_console() -> Console:
    global _stderr_console
    if _stderr_console is None:
        _reconfigure_utf8(sys.stderr)
        _stderr_console = Console(stderr=True, soft_wrap=False, highlight=False, theme=THEME)
    return _stderr_console


# Map logging levels onto :class:`Status`. The level-prefix renderer pulls
# icon + colour from the Status so glyphs live in exactly one place (the
# enum). DEBUG isn't a Status — we render it as a dim ``·`` separately.
_LEVEL_STATUS: dict[int, Status] = {
    logging.INFO: Status.INFO,
    logging.WARNING: Status.WARNING,
    logging.ERROR: Status.ERROR,
    logging.CRITICAL: Status.ERROR,
}


class RaitapRichHandler(RichHandler):
    """RichHandler with icon level-prefix and Panel rendering for multi-line warnings/errors."""

    _last_was_panel: bool = False

    def get_level_text(self, record: logging.LogRecord) -> Text:
        if record.levelno < logging.INFO:
            return Text("·", style=Style(dim=True))
        status = _LEVEL_STATUS.get(record.levelno, Status.INFO)
        shades = colour(status)
        # WARNING level-prefix uses ``light`` for legibility (ANSI 33 yellow
        # is brownish on a single glyph); other levels stay on ``base``.
        # CRITICAL adds bold on top of the ERROR base.
        if status is Status.WARNING:
            style = shades.light
        elif record.levelno >= logging.CRITICAL:
            style = shades.base + Style(bold=True)
        else:
            style = shades.base
        return Text(status.icon.rstrip(), style=style)

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
        status = Status.ERROR if record.levelno >= logging.ERROR else Status.WARNING
        body: Text = _linkify_message(message)
        chips: list[Text] = []
        label: str | None = None

        # RaitapError carries an explicit Diagnostic on the exception object,
        # which beats any heuristics we could apply to the formatted message.
        raitap_exc = _extract_raitap_error(record)
        if raitap_exc is not None and raitap_exc.diagnostic is not None:
            chips = diagnostic_chips(
                status,
                scope=(
                    raitap_exc.diagnostic.subsystem.capitalize()
                    if raitap_exc.diagnostic.subsystem
                    else type(raitap_exc).__name__
                ),
                src=(
                    f"{raitap_exc.diagnostic.file}:{raitap_exc.diagnostic.line}"
                    if raitap_exc.diagnostic.file
                    else ""
                ),
                diagnostic=raitap_exc.diagnostic,
            )
            self._print_frame(record, StatusFrame(status, body, label=label, chips=chips))
            return
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
            sub = diagnostic.subsystem if diagnostic else None
            scope = sub.capitalize() if sub else cat
            chips = diagnostic_chips(status, scope=scope, src=src, diagnostic=diagnostic)
            body = _linkify_message(warn_match.group("msg").strip())
        else:
            head_match = _LOGGER_HEADER_RE.match(message.strip())
            if head_match:
                head = head_match.group("head")
                shades = colour(status)
                # Prefer a traceback/path-derived Diagnostic so the chips can
                # carry a real file:line; fall back to a header-derived synth
                # for plain logger.warning("Foo: …") records that have no
                # exc_info and no informative pathname.
                resolved = _diagnostic_from_record(record)
                if resolved is None or (
                    resolved.subsystem is None and resolved.third_party_lib is None
                ):
                    resolved = Diagnostic(
                        subsystem=subsystem_from_str(head.lower()),
                        file="",
                        line=0,
                        third_party_lib=None,
                    )
                scope = resolved.subsystem.capitalize() if resolved.subsystem else head
                src = f"{resolved.file}:{resolved.line}" if resolved.file else ""
                derived = diagnostic_chips(status, scope=scope, src=src, diagnostic=resolved)
                chips = derived if derived else [chip(head, style=shades.light)]
                stripped = head_match.group("msg").strip()
                body = _linkify_message(
                    stripped[:1].upper() + stripped[1:] if stripped else stripped
                )

        if not chips:
            diagnostic = _diagnostic_from_record(record)
            if diagnostic is not None and (
                diagnostic.subsystem is not None or diagnostic.third_party_lib is not None
            ):
                scope = (
                    diagnostic.subsystem.capitalize()
                    if diagnostic.subsystem
                    else record.name.rsplit(".", 1)[-1].capitalize()
                )
                src = f"{diagnostic.file}:{diagnostic.line}" if diagnostic.file else ""
                chips = diagnostic_chips(status, scope=scope, src=src, diagnostic=diagnostic)

        self._print_frame(record, StatusFrame(status, body, label=label, chips=chips))

    def _print_frame(self, record: logging.LogRecord, frame: StatusFrame) -> None:
        try:
            # Two blank lines so the panel doesn't visually collide with a
            # progress bar or other Rich output that didn't leave trailing
            # whitespace (one blank line is sometimes consumed by the
            # progress renderer's final repaint).
            if not self._last_was_panel:
                self.console.print()
                self.console.print()
            self.console.print(frame.render())
            self.console.print()
            self._last_was_panel = True
        except Exception:
            # Never let logging crash the run — fall back to plain RichHandler.
            super().emit(record)


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


def _diagnostic_from_record(record: logging.LogRecord) -> Diagnostic | None:
    """Resolve a :class:`Diagnostic` for *any* log record.

    Walks ``record.exc_info`` traceback first (most informative — pinpoints the
    actual raising frame inside a raitap subsystem); falls back to the
    record's own ``pathname``/``lineno`` so non-exception ``logger.warning`` /
    ``logger.error`` calls still get scope and ``file:line`` chips. Returns
    ``None`` only when the record carries no usable location at all.
    """
    exc_info = record.exc_info
    if isinstance(exc_info, tuple) and len(exc_info) >= 3 and exc_info[2] is not None:
        return resolve_diagnostic_from_traceback(
            exc_info[2],
            default_file=record.pathname or "",
            default_line=record.lineno or 0,
        )
    if record.pathname:
        return resolve_diagnostic_from_path(record.pathname, record.lineno or 0)
    return None


def _extract_raitap_error(record: logging.LogRecord) -> RaitapError | None:
    """Return the :class:`RaitapError` attached to ``record.exc_info``, if any."""
    exc_info = record.exc_info
    if not exc_info:
        return None
    exc = exc_info[1] if isinstance(exc_info, tuple) else None
    if isinstance(exc, RaitapError):
        return exc
    return None


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


def diagnostic_chips(
    status: Status,
    *,
    scope: str,
    src: str,
    diagnostic: Diagnostic | None,
) -> list[Text]:
    """Return the chip sequence that decorates a status frame title.

    Layout depends on the audience:

    - **Dev install** (cloned repo): ``· <scope> · via <lib> · <path:line>``
      (``via <lib>`` only when a wrapped library is involved). ``path`` is
      clickable; all chips render in ``colour(status).light`` so the main
      label keeps the visual lead.
    - **Installed wheel, raitap-emitted**: ``· <scope> · View docs``,
      ``View docs`` linking to the subsystem documentation page.
    - **Installed wheel, third-party**: ``· <scope> · via <lib> · View docs``,
      link points to the frameworks-and-libraries doc page.
    - **Installed wheel, unclassified**: empty list.
    """
    shades = colour(status)
    sub = diagnostic.subsystem if diagnostic else None
    third_party = diagnostic.third_party_lib if diagnostic else None
    chips: list[Text] = []

    if is_dev_install():
        chips.append(chip(scope, style=shades.light))
        if third_party is not None:
            chips.append(chip(f"via {third_party.capitalize()}", style=shades.light))
        if src:
            chips.append(chip(src, style=shades.light, link=_src_to_uri(src)))
        return chips

    if sub is None:
        return chips
    chips.append(chip(scope, style=shades.light))
    if third_party is not None:
        chips.append(chip(f"via {third_party.capitalize()}", style=shades.light))
    url = docs_url(diagnostic) if diagnostic is not None else None
    if url is not None:
        chips.append(chip("View docs", style=shades.light, link=url, underline=True))
    return chips


def _format_value(value: Any, *, dot: bool = False, dot_style: Style | str | None = None) -> Text:
    """Render an arbitrary config value, gracefully handling ``None``/empty.

    ``dot=True`` prepends a colored ``●`` glyph (used for status fields).
    """
    success_style = colour(Status.SUCCESS).base
    if dot_style is None:
        dot_style = success_style
    if value is None or value == "" or value == [] or value == {}:
        return Text("—", style="dim")
    if isinstance(value, bool):
        if value:
            if dot:
                return Text.assemble(("● ", dot_style), ("on", success_style))
            return Text("on", style=success_style)
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


def print_complete_panel(duration: str) -> None:
    shades = colour(Status.SUCCESS)
    body = Text.assemble(
        (Status.SUCCESS.icon, shades.base),
        ("Assessment complete", shades.base + Style(bold=True)),
        ("    duration ", Style(dim=True)),
        (duration, Style(color="white")),
    )
    frame = StatusFrame(Status.SUCCESS, body)
    panel = frame.render()
    # Override default title (we already render icon+label in the body).
    panel.title = None
    get_console().print()
    get_console().print(panel)
    get_console().print()


def print_failure_panel(exc: BaseException, duration: str) -> None:
    shades = colour(Status.ERROR)
    body_pieces: list[tuple[str, Style] | tuple[str, str]] = [
        (Status.ERROR.icon, shades.base),
        ("Assessment failed", shades.base + Style(bold=True)),
        ("    after ", Style(dim=True)),
        (duration, Style(color="white")),
    ]
    chips: list[Text] = []
    label: str | None = None

    # Surface diagnostic chips when the failure carries a Diagnostic — same
    # affordance the rich handler renders for inline error records, but
    # printed at top-level by the Hydra entrypoint.
    if isinstance(exc, RaitapError) and exc.diagnostic is not None:
        label = "Failure"
        scope = (
            exc.diagnostic.subsystem.capitalize()
            if exc.diagnostic.subsystem
            else type(exc).__name__
        )
        src = f"{exc.diagnostic.file}:{exc.diagnostic.line}" if exc.diagnostic.file else ""
        chips = diagnostic_chips(Status.ERROR, scope=scope, src=src, diagnostic=exc.diagnostic)
        body_pieces.extend([("\n\n", ""), (str(exc), shades.base)])
        cause = exc.__cause__
        if cause is not None:
            body_pieces.extend(
                [
                    ("\n\n", ""),
                    (f"caused by {type(cause).__name__}: {cause}", Style(dim=True)),
                ]
            )
    else:
        body_pieces.extend([("\n\n", ""), (f"{type(exc).__name__}: {exc}", shades.base)])

    body = Text.assemble(*body_pieces)
    frame = StatusFrame(Status.ERROR, body, label=label, chips=chips)
    panel = frame.render()
    if label is None:
        # No diagnostic title — drop the auto-generated icon+label since the
        # body already opens with them.
        panel.title = None
    # Two blank lines so the panel separates cleanly from a progress bar
    # whose final repaint may eat the first newline.
    get_stderr_console().print()
    get_stderr_console().print()
    get_stderr_console().print(panel)
    get_stderr_console().print()


class _PercentColumn(ProgressColumn):
    def render(self, task: Any) -> Text:
        style = colour(Status.SUCCESS).base if task.finished else colour(Status.INFO).base
        pct = 0.0 if task.percentage is None else task.percentage
        return Text(f"{pct:>3.0f}%", style=style)


class _ElapsedColumn(ProgressColumn):
    def render(self, task: Any) -> Text:
        style = colour(Status.SUCCESS).base if task.finished else colour(Status.INFO).base
        elapsed = task.finished_time if task.finished else task.elapsed
        text = "-:--:--" if elapsed is None else str(timedelta(seconds=int(elapsed)))
        return Text(text, style=style)


def _make_progress() -> Progress:
    info = colour(Status.INFO).base
    success = colour(Status.SUCCESS).base
    return Progress(
        SpinnerColumn(style=info),
        TextColumn("{task.description}", style=info),
        BarColumn(complete_style=info, finished_style=success),
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
