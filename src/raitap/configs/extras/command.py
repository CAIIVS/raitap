"""Render the final ``uv sync``/``uv add`` argv and resolve install mode."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from raitap.utils.diagnostics import is_dev_install

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


def render_command(*, mode: Mode, extras: Iterable[str]) -> tuple[list[str], str]:
    """Return ``(argv, pretty_string)`` for the rendered uv command.

    ``extras`` is sorted and deduplicated for stable output.
    """
    sorted_extras = sorted(set(extras))

    if mode == "sync":
        argv = ["uv", "sync"]
        for extra in sorted_extras:
            argv.extend(["--extra", extra])
    elif mode == "add":
        argv = ["uv", "add"]
        if sorted_extras:
            argv.append(f"raitap[{','.join(sorted_extras)}]")
        else:
            argv.append("raitap")
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    return argv, " ".join(argv)
