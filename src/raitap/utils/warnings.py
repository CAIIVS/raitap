"""Warning infrastructure: origin classification, suppression, audience detection.

This module centralises warning-related logic so adapters (captum, shap, foolbox,
torchattacks, …) call named helpers instead of stdlib :mod:`warnings` directly,
and the console renderer can ask "where did this warning really come from, and
is the user looking at a cloned repo or a pip-installed wheel?".

Public surface:

- :class:`WarningOrigin` — typed result of frame-walking
- :func:`resolve_warn_origin` — walk frames to find raitap subsystem + third-party origin
- :func:`suppress_warning` — thin wrapper around :func:`warnings.filterwarnings`
- :func:`is_dev_install` — ``True`` when raitap is editable / cloned, ``False`` in site-packages
- :func:`docs_url` — subsystem-driven docs URL for an origin
"""

from __future__ import annotations

import re
import sys
import warnings
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

_SUBSYSTEM_RE = re.compile(r"raitap[\\/](?!src[\\/])(?P<sub>\w+)[\\/]")

_KNOWN_THIRD_PARTY_LIBS: Final[frozenset[str]] = frozenset(
    {"captum", "shap", "foolbox", "torchattacks"}
)

_DOCS_BASE: Final[str] = "https://caiivs.github.io/raitap"

# Subsystems that have dedicated docs pages.
_DOC_SUBSYSTEMS: Final[frozenset[str]] = frozenset(
    {"metrics", "transparency", "robustness", "data", "models", "reporting", "tracking"}
)

# Allowlist of legitimate raitap subsystem directory names. Anchoring against
# this set prevents false matches when raitap appears multiple times in a path
# (e.g. CI checkouts at ``/work/raitap/raitap/.venv/...``).
_KNOWN_SUBSYSTEMS: Final[frozenset[str]] = frozenset(
    {
        "configs",
        "data",
        "metrics",
        "models",
        "reporting",
        "robustness",
        "run",
        "tracking",
        "transparency",
        "utils",
    }
)


@dataclass(frozen=True)
class WarningOrigin:
    """Resolved origin of a warning.

    ``subsystem`` and ``third_party_lib`` are ``None`` when not classifiable.
    """

    subsystem: str | None
    file: str
    line: int
    third_party_lib: str | None


# In-order queue of origins resolved by :func:`_format_warning_compact`. Drained
# by the rich handler when it parses a warning panel. Single-process / mostly
# single-threaded use; emit order matches warn order.
_PENDING_ORIGINS: deque[WarningOrigin] = deque()


def _detect_third_party(path: str) -> str | None:
    normalized = path.replace("\\", "/")
    for lib in _KNOWN_THIRD_PARTY_LIBS:
        if f"/{lib}/" in normalized:
            return lib
    return None


def _classify_subsystem(path: str) -> str | None:
    match = _SUBSYSTEM_RE.search(path)
    if match is None:
        return None
    sub = match.group("sub")
    return sub if sub in _KNOWN_SUBSYSTEMS else None


def resolve_warn_origin(default_file: str, default_line: int) -> WarningOrigin:
    """Walk the live call stack to find the actual warning origin.

    Returns a :class:`WarningOrigin` with:

    - ``file`` / ``line``: first frame inside ``raitap/<subsystem>/`` (so users see
      a location they can act on), falling back to ``default_file:default_line``.
    - ``subsystem``: the matching ``<subsystem>``, or ``None``.
    - ``third_party_lib``: name of a known wrapped library if a frame in the
      stack lives inside it; otherwise ``None``. This is checked against the
      *initial* default file as well as inner frames.
    """
    third_party = _detect_third_party(default_file)
    rai_path: str | None = None
    rai_line: int = default_line
    rai_sub: str | None = None

    frame: object = sys._getframe()
    while frame is not None:
        f_code = getattr(frame, "f_code", None)
        if f_code is None:
            frame = getattr(frame, "f_back", None)
            continue
        path = f_code.co_filename
        normalized = path.replace("\\", "/")
        if third_party is None:
            third_party = _detect_third_party(path)
        if rai_path is None and "/raitap/" in normalized:
            sub = _classify_subsystem(path)
            if sub is not None and "/raitap/utils/" not in normalized:
                rai_path = path
                rai_line = getattr(frame, "f_lineno", default_line)
                rai_sub = sub
        frame = getattr(frame, "f_back", None)

    if rai_path is None:
        return WarningOrigin(
            subsystem=None,
            file=default_file,
            line=default_line,
            third_party_lib=third_party,
        )
    return WarningOrigin(
        subsystem=rai_sub,
        file=rai_path,
        line=rai_line,
        third_party_lib=third_party,
    )


def suppress_warning(
    *,
    message: str,
    category: type[Warning] = UserWarning,
    module: str = "",
) -> None:
    """Register a runtime ``warnings.filterwarnings("ignore", ...)`` filter.

    Adapters call this at import time to silence known-noise warnings their
    wrapped library emits. The match arguments mirror :func:`warnings.filterwarnings`.
    """
    warnings.filterwarnings("ignore", message=message, category=category, module=module)


@lru_cache(maxsize=1)
def is_dev_install() -> bool:
    """Return ``True`` when raitap appears to run from a cloned/editable checkout.

    Heuristic: the package's ``__file__`` lives outside any ``site-packages``
    segment. Editable installs (``uv pip install -e``) point ``__file__`` at the
    source tree, so they correctly report ``True`` and contributors keep the
    full ``path:line`` subheader.
    """
    try:
        import raitap

        location = (raitap.__file__ or "").replace("\\", "/")
    except Exception:
        return False
    if not location:
        return False
    return "/site-packages/" not in location


def docs_url(origin: WarningOrigin) -> str | None:
    """Return a documentation URL for an origin, or ``None`` if unclassified."""
    sub = origin.subsystem
    if sub is None:
        return None
    if sub not in _DOC_SUBSYSTEMS:
        return None
    if origin.third_party_lib is not None:
        return f"{_DOCS_BASE}/modules/{sub}/frameworks-and-libraries.html"
    return f"{_DOCS_BASE}/modules/{sub}/"


def _push_origin(origin: WarningOrigin) -> None:
    """Stash an origin for the rich handler to drain."""
    _PENDING_ORIGINS.append(origin)


def _pop_origin() -> WarningOrigin | None:
    """Pop the next stashed origin, or ``None`` if the queue is empty."""
    if not _PENDING_ORIGINS:
        return None
    return _PENDING_ORIGINS.popleft()


def _clear_origins() -> None:
    """Test helper: reset the pending-origin queue."""
    _PENDING_ORIGINS.clear()


__all__ = [
    "WarningOrigin",
    "docs_url",
    "is_dev_install",
    "resolve_warn_origin",
    "suppress_warning",
]
