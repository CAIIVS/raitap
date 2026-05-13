"""Reject selected extras whose deps cannot install on the current platform.

A raitap config that requires (say) ``marabou`` on Windows will install the
``marabou`` extra cleanly, but its ``maraboupy`` requirement carries a
``sys_platform != 'win32'`` marker and silently drops out. The run then
crashes later when ``MarabouAssessor`` tries to ``import maraboupy``.

This module hard-fails *before* uv runs: if any requirement of a selected
extra has a marker that evaluates ``False`` for every candidate Python on the
current platform, raise :class:`ExtraUnavailableError` and tell the user to
adjust the config or switch host.
"""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING

from packaging.requirements import Requirement

from raitap.configs.extras.python_version import _base_env, _candidate_pythons, _env_for

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class ExtraUnavailableError(RuntimeError):
    """Raised when a selected extra cannot install on the current platform."""


def check_platform_availability(
    pyproject_path: Path,
    extras: Iterable[str],
) -> None:
    """Raise :class:`ExtraUnavailableError` if any selected extra is unreachable.

    A requirement counts as unreachable when its PEP 508 marker is ``False``
    for the current platform across every candidate Python permitted by
    ``requires-python``. Markers that pass for at least one candidate are
    fine — the Python pin (see :mod:`raitap.configs.extras.python_version`)
    will select a compatible interpreter.
    """
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    optional = project.get("optional-dependencies", {})
    candidates = _candidate_pythons(project.get("requires-python"))
    if not candidates:
        return

    base = _base_env()
    failures: list[str] = []

    for extra in sorted(set(extras)):
        for req_str in optional.get(extra, []):
            req = Requirement(req_str)
            if req.marker is None:
                continue
            reachable = any(req.marker.evaluate({**base, **_env_for(cand)}) for cand in candidates)
            if not reachable:
                failures.append(
                    f"extra '{extra}' requires '{req.name}' but its marker "
                    f"'{req.marker}' rejects {base.get('sys_platform', '?')} "
                    "on every supported Python"
                )

    if failures:
        joined = "\n  - " + "\n  - ".join(failures)
        raise ExtraUnavailableError(
            "Inferred extras include packages unavailable on this platform. "
            "Fix the config (drop the offending block) or switch host."
            f"{joined}"
        )
