"""Pick the right Python interpreter for a set of inferred extras.

Some raitap extras pin Python (e.g. ``marabou`` ships cp311 wheels only on
Linux/macOS). When such an extra is selected, ``uv sync``/``uv add`` should
run with ``-p X.Y`` so uv resolves into a compatible interpreter rather than
the host's default.

The logic walks ``[project.optional-dependencies]`` in ``pyproject.toml``,
parses each requirement's PEP 508 marker via :mod:`packaging`, and finds the
highest Python within ``requires-python`` that satisfies every marker that
imposes a Python constraint on the current platform.

A requirement marker that evaluates ``False`` on the current platform for
*every* candidate Python is treated as platform-skipped (uv will simply not
install the package on this OS) and contributes no Python constraint.
"""

from __future__ import annotations

import platform as _platform
import sys
import tomllib
from typing import TYPE_CHECKING

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

_DEFAULT_CANDIDATES = ((3, 11), (3, 12), (3, 13))


def _candidate_pythons(requires_python: str | None) -> list[tuple[int, int]]:
    if not requires_python:
        return list(_DEFAULT_CANDIDATES)
    spec = SpecifierSet(requires_python)
    return [c for c in _DEFAULT_CANDIDATES if spec.contains(f"{c[0]}.{c[1]}.0")]


def _base_env() -> dict[str, str]:
    """Marker env without ``python_version``/``python_full_version``."""
    vi = sys.version_info
    return {
        "implementation_name": sys.implementation.name,
        "implementation_version": f"{vi.major}.{vi.minor}.{vi.micro}",
        "os_name": _platform.system().lower() or "",
        "platform_machine": _platform.machine(),
        "platform_release": _platform.release(),
        "platform_system": _platform.system(),
        "platform_version": _platform.version(),
        "sys_platform": sys.platform,
        "platform_python_implementation": _platform.python_implementation(),
    }


def _env_for(candidate: tuple[int, int]) -> dict[str, str]:
    env = _base_env()
    env["python_version"] = f"{candidate[0]}.{candidate[1]}"
    env["python_full_version"] = f"{candidate[0]}.{candidate[1]}.0"
    return env


def pick_python_version(
    pyproject_path: Path,
    extras: Iterable[str],
) -> str | None:
    """Return the highest Python ``X.Y`` compatible with ``extras``.

    Returns ``None`` when the extras impose no Python constraint beyond
    ``requires-python``. Returns a ``"X.Y"`` string (e.g. ``"3.11"``) when at
    least one extra has a Python-pinning marker on this platform.
    """
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    optional = project.get("optional-dependencies", {})
    requires_python = project.get("requires-python")

    candidates = _candidate_pythons(requires_python)
    if not candidates:
        return None

    pinned = False
    viable = set(candidates)

    for extra in set(extras):
        for req_str in optional.get(extra, []):
            req = Requirement(req_str)
            if req.marker is None:
                continue
            allowed_for_marker: set[tuple[int, int]] = set()
            for cand in candidates:
                if req.marker.evaluate(_env_for(cand)):
                    allowed_for_marker.add(cand)
            if not allowed_for_marker:
                # Marker rejects every candidate on this platform — package is
                # platform-skipped, not a Python constraint.
                continue
            if allowed_for_marker == set(candidates):
                # Marker passes everywhere relevant; no Python pin from this req.
                continue
            pinned = True
            viable &= allowed_for_marker

    if not pinned or not viable:
        return None

    best = sorted(viable)[-1]
    return f"{best[0]}.{best[1]}"
