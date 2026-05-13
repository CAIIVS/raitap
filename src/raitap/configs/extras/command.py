"""Render the final ``uv sync``/``uv add`` argv and resolve install mode."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from raitap.utils.diagnostics import is_dev_install

if TYPE_CHECKING:
    from collections.abc import Iterable

Mode = Literal["sync", "add"]
ModeRequest = Literal["auto", "sync", "add"]


def select_mode(requested: ModeRequest) -> Mode:
    """Resolve the user's --mode flag to a concrete mode.

    ``auto`` yields ``sync`` for dev checkouts, ``add`` otherwise.
    """
    if requested == "auto":
        return "sync" if is_dev_install() else "add"
    if requested in ("sync", "add"):
        return requested
    raise ValueError(f"Unknown mode: {requested!r}")


def render_command(
    *,
    mode: Mode,
    extras: Iterable[str],
    python_version: str | None = None,
) -> tuple[list[str], str]:
    """Return ``(argv, pretty_string)`` for the rendered uv command.

    ``extras`` is sorted and deduplicated for stable output. When
    ``python_version`` is set, ``-p X.Y`` is inserted after the subcommand
    so uv resolves into that interpreter.
    """
    sorted_extras = sorted(set(extras))

    if mode == "sync":
        argv = ["uv", "sync"]
    elif mode == "add":
        argv = ["uv", "add"]
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    if python_version is not None:
        argv.extend(["-p", python_version])

    if mode == "sync":
        for extra in sorted_extras:
            argv.extend(["--extra", extra])
    else:  # add
        argv.append(f"raitap[{','.join(sorted_extras)}]" if sorted_extras else "raitap")

    return argv, " ".join(argv)
