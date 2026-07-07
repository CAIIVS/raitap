"""Diagnostic infrastructure shared by warnings and errors.

A :class:`Diagnostic` describes *where* an in-flight problem comes from inside
raitap (module + source location, plus an optional third-party library
flag). The same dataclass is reused for warnings (populated from frame walks
in :mod:`raitap.utils.log`) and, in future, for errors raised by the
robustness/transparency adapters (issue #22), where it will be populated from
an exception's traceback.

Public surface:

- :class:`Diagnostic` — typed origin record (file/line/module/third-party)
- :class:`Module` — type-safe enum of raitap module directory names
- :func:`resolve_diagnostic_from_frames` — walk live frames to find the raitap module
- :func:`is_dev_install` — ``True`` for cloned/editable checkouts, ``False`` for installed wheels
- :func:`docs_url` — module-driven docs URL for a diagnostic

Third-party library detection lives here, but the *list* of wrapped libraries
does not: each ``@register_*_adapter(..., library="...")`` decoration auto-
populates :data:`raitap._adapters.THIRD_PARTY_LIBS` at module-load time,
keyed by adapter family. :func:`_third_party_libs` aggregates the dict
lazily after importing every family root. Adding a new wrapped library is
the same as adding the adapter — no edit here.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from types import TracebackType  # noqa: TC003 — runtime import: keeps public hints resolvable
from typing import Final

_SUBSYSTEM_RE = re.compile(r"raitap[\\/](?!src[\\/])(?P<sub>\w+)[\\/]")

_DOCS_BASE: Final[str] = "https://caiivs.github.io/raitap"


class Module(StrEnum):
    """Type-safe enum of legitimate raitap module directory names.

    Anchoring origin classification to this enum prevents false matches when
    ``raitap`` appears multiple times in a path (e.g. CI checkouts at
    ``/work/raitap/raitap/.venv/...``). ``StrEnum`` so members are interchangeable
    with their string values for logging, dataclass fields, and existing tests.
    """

    cli = "cli"
    configs = "configs"
    data = "data"
    deps = "deps"
    metrics = "metrics"
    models = "models"
    pipeline = "pipeline"
    reporting = "reporting"
    robustness = "robustness"
    tracking = "tracking"
    transparency = "transparency"
    utils = "utils"


def module_from_str(name: str) -> Module | None:
    """Return the :class:`Module` matching ``name``, or ``None`` if unknown."""
    try:
        return Module(name)
    except ValueError:
        return None


# Modules that are infrastructure rather than user-facing modules and so
# have no dedicated docs page. All other ``Module`` members do.
_NO_DOC_MODULESS: Final[frozenset[Module]] = frozenset(
    {Module.cli, Module.configs, Module.deps, Module.pipeline, Module.utils}
)
_DOC_SUBSYSTEMS: Final[frozenset[Module]] = frozenset(Module) - _NO_DOC_MODULESS


@dataclass(frozen=True)
class Diagnostic:
    """Resolved origin of a warning or error.

    ``module`` and ``third_party_lib`` are ``None`` when not classifiable.
    """

    module: Module | None
    file: str
    line: int
    third_party_lib: str | None


def _third_party_libs() -> frozenset[str]:
    """Aggregate wrapped third-party libraries across every adapter family.

    Reads :data:`raitap._adapters.THIRD_PARTY_LIBS`, which
    :func:`raitap._adapters._register_core` populates from each
    ``@register_*_adapter(..., library="...")`` decoration.
    :func:`raitap.configs.register_configs` is invoked first (idempotent) so
    every in-tree family is imported and third-party plugins are discovered —
    firing those decorations without ``utils`` hardcoding which packages exist.
    """
    from raitap.configs import register_configs

    register_configs()

    from raitap._adapters import THIRD_PARTY_LIBS as _LIBS_BY_GROUP

    aggregated: set[str] = set()
    for libs in _LIBS_BY_GROUP.values():
        aggregated |= libs
    return frozenset(aggregated)


def _detect_third_party(path: str) -> str | None:
    normalized = path.replace("\\", "/")
    for lib in _third_party_libs():
        if f"/{lib}/" in normalized:
            return lib
    return None


def _classify_module(path: str) -> Module | None:
    match = _SUBSYSTEM_RE.search(path)
    if match is None:
        return None
    return module_from_str(match.group("sub"))


def resolve_diagnostic_from_frames(default_file: str, default_line: int) -> Diagnostic:
    """Walk the live call stack to find the actual diagnostic origin.

    Returns a :class:`Diagnostic` with:

    - ``file`` / ``line``: first frame inside ``raitap/<module>/`` (so users see
      a location they can act on), falling back to ``default_file:default_line``.
    - ``module``: the matching ``<module>``, or ``None``.
    - ``third_party_lib``: name of a known wrapped library if a frame in the
      stack lives inside it; otherwise ``None``. Checked against the *initial*
      default file as well as inner frames.
    """
    third_party = _detect_third_party(default_file)
    rai_path: str | None = None
    rai_line: int = default_line
    rai_sub: Module | None = None

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
            sub = _classify_module(path)
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
            module=None,
            file=default_file,
            line=default_line,
            third_party_lib=third_party,
        )
    return Diagnostic(
        module=rai_sub,
        file=rai_path,
        line=rai_line,
        third_party_lib=third_party,
    )


def resolve_diagnostic_from_traceback(
    tb: TracebackType | None,
    default_file: str = "",
    default_line: int = 0,
) -> Diagnostic:
    """Walk an exception traceback to find the deepest raitap module frame.

    Unlike :func:`resolve_diagnostic_from_frames`, the traceback survives the
    ``except`` handler, so we can still classify origins after the live frames
    have been unwound (e.g. inside a ``logger.exception`` handler at emit time).
    Picks the *deepest* matching module frame so the chip points at the
    actual raising site, not the entry point.
    """
    third_party: str | None = None
    rai_path: str | None = None
    rai_line: int = default_line
    rai_sub: Module | None = None

    cur = tb
    while cur is not None:
        path = cur.tb_frame.f_code.co_filename
        normalized = path.replace("\\", "/")
        if third_party is None:
            third_party = _detect_third_party(path)
        if "/raitap/" in normalized and "/raitap/utils/" not in normalized:
            sub = _classify_module(path)
            if sub is not None:
                rai_path = path
                rai_line = cur.tb_lineno
                rai_sub = sub
        cur = cur.tb_next

    if rai_path is None:
        return Diagnostic(
            module=None,
            file=default_file,
            line=default_line,
            third_party_lib=third_party,
        )
    return Diagnostic(
        module=rai_sub,
        file=rai_path,
        line=rai_line,
        third_party_lib=third_party,
    )


def resolve_diagnostic_from_path(path: str, line: int) -> Diagnostic:
    """Classify a single ``path:line`` location without walking any stack.

    Useful for log records that carry their emission site via
    ``record.pathname`` / ``record.lineno`` but no exception traceback. Returns
    a :class:`Diagnostic` with ``module``/``file`` populated when the path
    sits inside a non-``utils`` raitap module; otherwise the third-party
    flag may still be set if the path lives inside a wrapped library.
    """
    normalized = path.replace("\\", "/")
    sub: Module | None = None
    if "/raitap/" in normalized and "/raitap/utils/" not in normalized:
        sub = _classify_module(path)
    third_party = _detect_third_party(path)
    return Diagnostic(
        module=sub,
        file=path if sub is not None else "",
        line=line if sub is not None else 0,
        third_party_lib=third_party,
    )


@lru_cache(maxsize=1)
def is_dev_install() -> bool:
    """Return ``True`` when raitap is being run from its OWN cloned checkout.

    Two conditions must hold:

    1. The package's ``__file__`` lives outside any ``site-packages`` /
       ``dist-packages`` segment (editable / source layout).
    2. The current working directory's ``pyproject.toml`` declares
       ``project.name = "raitap"``.

    Condition 1 alone is not enough: a consumer project that pulls raitap via
    ``tool.uv.sources = { path = "..", editable = true }`` also points raitap's
    ``__file__`` at a source tree, but the *current* project is the consumer,
    not raitap itself. Bootstrap's "dev" auto-sync targets the cwd pyproject,
    so dev-detection must check cwd too.
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
    if "/site-packages/" in location or "/dist-packages/" in location:
        return False

    # Condition 2: cwd's pyproject must be raitap's.
    from pathlib import Path

    cwd_pyproject = Path.cwd() / "pyproject.toml"
    if not cwd_pyproject.exists():
        return False
    try:
        import tomllib

        with cwd_pyproject.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception:
        return False
    return data.get("project", {}).get("name") == "raitap"


def docs_url(diagnostic: Diagnostic) -> str | None:
    """Return a documentation URL for a diagnostic, or ``None`` if unclassified."""
    sub = diagnostic.module
    if sub is None:
        return None
    if sub not in _DOC_SUBSYSTEMS:
        return None
    if diagnostic.third_party_lib is not None:
        return f"{_DOCS_BASE}/modules/{sub}/frameworks-and-libraries.html"
    return f"{_DOCS_BASE}/modules/{sub}/"


__all__ = [
    "Diagnostic",
    "Module",
    "docs_url",
    "is_dev_install",
    "module_from_str",
    "resolve_diagnostic_from_frames",
    "resolve_diagnostic_from_path",
    "resolve_diagnostic_from_traceback",
]
