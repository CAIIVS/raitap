"""Best-effort static scan of adapter ``extra=`` declarations.

Walks the installed raitap source tree with :mod:`ast` and harvests every
``@register_<family>_adapter(..., extra="bar")`` (or
``@register_<family>_visualiser(..., extra="bar")``) decorator above a
``class`` def, without importing the module (and therefore without pulling
the wrapped third-party library).

Used by :func:`raitap.deps.inference._extra_for_target` so the deps
bootstrap can resolve ``_target_`` → extra mappings in partial-extras
venvs — i.e. before the very libraries it is about to install have been
installed. The runtime ``ADAPTER_EXTRAS`` dict populated by
:func:`raitap._adapters._register_core` is still the authoritative source
whenever it is populated; this scanner is a safety net for
``raitap.run(..., auto_install=True)`` and the CLI bootstrap.

Preserves the "adding a new adapter = single-file add" invariant:
nothing here is hand-maintained, and a new adapter's
``@register_*_adapter(..., extra="...")`` is picked up automatically.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path


def _decorator_name(deco: ast.expr) -> str | None:
    """Return the bare name of a decorator call (e.g. ``register_metrics_adapter``)
    regardless of whether it was imported as ``register_metrics_adapter`` or
    accessed via ``module.register_metrics_adapter``."""
    if not isinstance(deco, ast.Call):
        return None
    func = deco.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _is_register_decorator(name: str | None) -> bool:
    """Match every family registration decorator: ``register_transparency_adapter``,
    ``register_robustness_adapter``, ``register_metrics_adapter``,
    ``register_reporter``, ``register_tracker``,
    ``register_transparency_visualiser``, ``register_robustness_visualiser``.
    All share the ``register_`` prefix — broader match keeps this future-proof
    when a new family adds its own decorator."""
    return name is not None and name.startswith("register_")


@lru_cache(maxsize=1)
def scan_adapter_extras() -> dict[str, str]:
    """Return ``{class_name: extra}`` harvested from raitap's source tree."""
    import raitap

    root = Path(raitap.__file__).resolve().parent
    found: dict[str, str] = {}
    for path in root.rglob("*.py"):
        # Tests can legitimately declare ``@register_*_adapter(..., extra="…")`` —
        # they are not real adapters and should not pollute the map.
        if "tests" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, OSError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for deco in node.decorator_list:
                if not _is_register_decorator(_decorator_name(deco)):
                    continue
                assert isinstance(deco, ast.Call)  # narrowed by _decorator_name
                for kw in deco.keywords:
                    if (
                        kw.arg == "extra"
                        and isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, str)
                    ):
                        found[node.name] = kw.value.value
                        break
                if node.name in found:
                    break
    return found
