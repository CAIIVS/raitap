"""Diagnostic infrastructure shared by warnings and errors.

A :class:`Diagnostic` describes *where* an in-flight problem comes from inside
raitap (subsystem + source location, plus an optional third-party library
flag). The same dataclass is reused for warnings (populated from frame walks
in :mod:`raitap.utils.log`) and, in future, for errors raised by the
robustness/transparency adapters (issue #22), where it will be populated from
an exception's traceback.

Public surface:

- :class:`Diagnostic` — typed origin record (file/line/subsystem/third-party)
- :class:`Subsystem` — type-safe enum of raitap subsystem directory names
- :func:`resolve_diagnostic_from_frames` — walk live frames to find the raitap subsystem
- :func:`is_dev_install` — ``True`` for cloned/editable checkouts, ``False`` for installed wheels
- :func:`docs_url` — subsystem-driven docs URL for a diagnostic

Third-party library detection lives here, but the *list* of wrapped libraries
does not: each subsystem package (``raitap.transparency``, ``raitap.robustness``,
…) exposes a ``THIRD_PARTY_LIBS`` constant, aggregated lazily by
:func:`_third_party_libs`. Adding a new wrapped library is a one-line edit in
the relevant subsystem ``__init__``, not here.
"""

from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from types import TracebackType

_SUBSYSTEM_RE = re.compile(r"raitap[\\/](?!src[\\/])(?P<sub>\w+)[\\/]")

# Subsystem packages that wrap third-party libraries and expose a
# ``THIRD_PARTY_LIBS`` constant. Aggregated lazily by :func:`_third_party_libs`
# so ``utils`` stays free of an upward dependency on these packages.
_THIRD_PARTY_LIB_PROVIDERS: Final[tuple[str, ...]] = (
    "raitap.transparency",
    "raitap.robustness",
)

_DOCS_BASE: Final[str] = "https://caiivs.github.io/raitap"


class Subsystem(StrEnum):
    """Type-safe enum of legitimate raitap subsystem directory names.

    Anchoring origin classification to this enum prevents false matches when
    ``raitap`` appears multiple times in a path (e.g. CI checkouts at
    ``/work/raitap/raitap/.venv/...``). ``StrEnum`` so members are interchangeable
    with their string values for logging, dataclass fields, and existing tests.
    """

    configs = "configs"
    data = "data"
    metrics = "metrics"
    models = "models"
    reporting = "reporting"
    robustness = "robustness"
    run = "run"
    tracking = "tracking"
    transparency = "transparency"
    utils = "utils"


def subsystem_from_str(name: str) -> Subsystem | None:
    """Return the :class:`Subsystem` matching ``name``, or ``None`` if unknown."""
    try:
        return Subsystem(name)
    except ValueError:
        return None


# Subsystems that are infrastructure rather than user-facing modules and so
# have no dedicated docs page. All other ``Subsystem`` members do.
_NO_DOC_SUBSYSTEMS: Final[frozenset[Subsystem]] = frozenset(
    {Subsystem.configs, Subsystem.run, Subsystem.utils}
)
_DOC_SUBSYSTEMS: Final[frozenset[Subsystem]] = frozenset(Subsystem) - _NO_DOC_SUBSYSTEMS


@dataclass(frozen=True)
class Diagnostic:
    """Resolved origin of a warning or error.

    ``subsystem`` and ``third_party_lib`` are ``None`` when not classifiable.
    """

    subsystem: Subsystem | None
    file: str
    line: int
    third_party_lib: str | None


@lru_cache(maxsize=1)
def _third_party_libs() -> frozenset[str]:
    """Aggregate ``THIRD_PARTY_LIBS`` from each subsystem package.

    Imported lazily so ``utils.diagnostics`` doesn't pay the cost (or pull in
    optional dependencies) until the first warning needs classifying. Cached
    after the first successful aggregation.
    """
    libs: set[str] = set()
    for module_name in _THIRD_PARTY_LIB_PROVIDERS:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        libs |= getattr(module, "THIRD_PARTY_LIBS", frozenset())
    return frozenset(libs)


def _detect_third_party(path: str) -> str | None:
    normalized = path.replace("\\", "/")
    for lib in _third_party_libs():
        if f"/{lib}/" in normalized:
            return lib
    return None


def _classify_subsystem(path: str) -> Subsystem | None:
    match = _SUBSYSTEM_RE.search(path)
    if match is None:
        return None
    return subsystem_from_str(match.group("sub"))


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
    rai_sub: Subsystem | None = None

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


def resolve_diagnostic_from_traceback(
    tb: TracebackType | None,
    default_file: str = "",
    default_line: int = 0,
) -> Diagnostic:
    """Walk an exception traceback to find the deepest raitap subsystem frame.

    Unlike :func:`resolve_diagnostic_from_frames`, the traceback survives the
    ``except`` handler, so we can still classify origins after the live frames
    have been unwound (e.g. inside a ``logger.exception`` handler at emit time).
    Picks the *deepest* matching subsystem frame so the chip points at the
    actual raising site, not the entry point.
    """
    third_party: str | None = None
    rai_path: str | None = None
    rai_line: int = default_line
    rai_sub: Subsystem | None = None

    cur = tb
    while cur is not None:
        path = cur.tb_frame.f_code.co_filename
        normalized = path.replace("\\", "/")
        if third_party is None:
            third_party = _detect_third_party(path)
        if "/raitap/" in normalized and "/raitap/utils/" not in normalized:
            sub = _classify_subsystem(path)
            if sub is not None:
                rai_path = path
                rai_line = cur.tb_lineno
                rai_sub = sub
        cur = cur.tb_next

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


def resolve_diagnostic_from_path(path: str, line: int) -> Diagnostic:
    """Classify a single ``path:line`` location without walking any stack.

    Useful for log records that carry their emission site via
    ``record.pathname`` / ``record.lineno`` but no exception traceback. Returns
    a :class:`Diagnostic` with ``subsystem``/``file`` populated when the path
    sits inside a non-``utils`` raitap subsystem; otherwise the third-party
    flag may still be set if the path lives inside a wrapped library.
    """
    normalized = path.replace("\\", "/")
    sub: Subsystem | None = None
    if "/raitap/" in normalized and "/raitap/utils/" not in normalized:
        sub = _classify_subsystem(path)
    third_party = _detect_third_party(path)
    return Diagnostic(
        subsystem=sub,
        file=path if sub is not None else "",
        line=line if sub is not None else 0,
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
    "Subsystem",
    "docs_url",
    "is_dev_install",
    "resolve_diagnostic_from_frames",
    "resolve_diagnostic_from_path",
    "resolve_diagnostic_from_traceback",
    "subsystem_from_str",
]
