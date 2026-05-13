"""Render the final ``uv sync`` / ``uv add`` / ``pip install`` argv."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Literal

from raitap.utils.diagnostics import is_dev_install

if TYPE_CHECKING:
    from collections.abc import Iterable

Mode = Literal["sync", "add", "pip"]
ModeRequest = Literal["auto", "sync", "add", "pip"]


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
        if python_version is not None:
            argv.extend(["-p", python_version])
        for extra in sorted_extras:
            argv.extend(["--extra", extra])
    elif mode == "add":
        argv = ["uv", "add"]
        if python_version is not None:
            argv.extend(["-p", python_version])
        argv.append(f"raitap[{','.join(sorted_extras)}]" if sorted_extras else "raitap")
    elif mode == "pip":
        # Use the running interpreter's own pip so the install lands in the
        # correct site-packages (venv vs system) — pip cannot switch
        # interpreters, so ``python_version`` is ignored here. Callers must
        # validate the host Python before selecting this mode.
        argv = [sys.executable, "-m", "pip", "install"]
        argv.append(f"raitap[{','.join(sorted_extras)}]" if sorted_extras else "raitap")
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    return argv, " ".join(argv)
