"""Diagnostic infrastructure shared by warnings and errors.

A :class:`Diagnostic` describes *where* an in-flight problem comes from inside
raitap (subsystem + source location, plus an optional third-party library
flag). The same dataclass is reused for warnings (populated from frame walks
in :mod:`raitap.utils.warnings`) and, in future, for errors raised by the
robustness/transparency adapters (issue #22), where it will be populated from
an exception's traceback.

Public surface:

- :class:`Diagnostic` — typed origin record (file/line/subsystem/third-party)
- :func:`resolve_diagnostic_from_frames` — walk live frames to find the raitap subsystem
- :func:`is_dev_install` — ``True`` for cloned/editable checkouts, ``False`` for installed wheels
- :func:`docs_url` — subsystem-driven docs URL for a diagnostic
"""

from __future__ import annotations

import re
import sys
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
class Diagnostic:
    """Resolved origin of a warning or error.

    ``subsystem`` and ``third_party_lib`` are ``None`` when not classifiable.
    """

    subsystem: str | None
    file: str
    line: int
    third_party_lib: str | None


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


def resolve_diagnostic_from_frames(default_file: str, default_line: int) -> Diagnostic:
    """Walk the live call stack to find the actual diagnostic origin.

    Returns a :class:`Diagnostic` with:

    - ``file`` / ``line``: first frame inside ``raitap/<subsystem>/`` (so users see
      a location they can act on), falling back to ``default_file:default_line``.
    - ``subsystem``: the matching ``<subsystem>``, or ``None``.
    - ``third_party_lib``: name of a known wrapped library if a frame in the
      stack lives inside it; otherwise ``None``. Checked against the *initial*
      default file as well as inner frames.
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
        # Both pieces of information resolved — no need to keep walking the
        # rest of the (potentially deep) stack just to confirm.
        if rai_path is not None and third_party is not None:
            break
        frame = getattr(frame, "f_back", None)

    if rai_path is None:
        return Diagnostic(
            subsystem=None,
            file=default_file,
            line=default_line,
            third_party_lib=third_party,
        )
    return Diagnostic(
        subsystem=rai_sub,
        file=rai_path,
        line=rai_line,
        third_party_lib=third_party,
    )


@lru_cache(maxsize=1)
def is_dev_install() -> bool:
    """Return ``True`` when raitap appears to run from a cloned/editable checkout.

    Heuristic: the package's ``__file__`` lives outside any ``site-packages`` /
    ``dist-packages`` segment (case-insensitive). Editable installs (``uv pip
    install -e``) point ``__file__`` at the source tree, so they correctly
    report ``True`` and contributors keep the full ``path:line`` subheader.
    """
    try:
        import raitap

        location = (raitap.__file__ or "").replace("\\", "/").lower()
    except Exception:
        return False
    if not location:
        return False
    # ``site-packages`` (PyPI / venv) and ``dist-packages`` (Debian/Ubuntu) both
    # indicate an installed wheel. Case-insensitive to handle Windows ``Lib``
    # variants and odd casing in custom layouts.
    return "/site-packages/" not in location and "/dist-packages/" not in location


def docs_url(diagnostic: Diagnostic) -> str | None:
    """Return a documentation URL for a diagnostic, or ``None`` if unclassified."""
    sub = diagnostic.subsystem
    if sub is None:
        return None
    if sub not in _DOC_SUBSYSTEMS:
        return None
    if diagnostic.third_party_lib is not None:
        return f"{_DOCS_BASE}/modules/{sub}/frameworks-and-libraries.html"
    return f"{_DOCS_BASE}/modules/{sub}/"


__all__ = [
    "Diagnostic",
    "docs_url",
    "is_dev_install",
    "resolve_diagnostic_from_frames",
]
