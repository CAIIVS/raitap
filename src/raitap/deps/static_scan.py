"""Best-effort static scan of adapter ``extra=`` declarations.

Walks the installed raitap source tree with :mod:`ast` and harvests every
family decorator (``@<family>_adapter(...)`` / ``@<family>_visualiser(...)``)
above a ``class`` def, without importing the module (and therefore without
pulling the wrapped third-party library).

Mirrors :func:`raitap._adapters._register_core`'s defaulting: when the
decorator omits ``extra=``, the runtime defaults it to ``registry_name``, so
the scanner does the same.

Used by :func:`raitap.deps.inference._extra_for_target` so the deps
bootstrap can resolve ``_target_`` → extra mappings in partial-extras
venvs — i.e. before the very libraries it is about to install have been
installed. The runtime ``ADAPTER_EXTRAS`` dict populated by
:func:`raitap._adapters._register_core` is still the authoritative source
whenever it is populated; this scanner is a safety net for
``raitap.run(..., auto_install_deps=True)`` and the CLI bootstrap.

Preserves the "adding a new adapter = single-file add" invariant:
nothing here is hand-maintained, and a new adapter's
``@<family>_adapter(..., extra="...")`` is picked up automatically.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

# In-tree adapters decorate with the bare family decorator imported directly
# from its ``registration`` module (e.g. ``@metrics_adapter(...)``). The public
# ``@adapters.<family>`` facade is for external plugin authors and is not used
# in-tree (it would create an import cycle), so the scanner matches bare names.
_ADAPTER_DECORATORS = frozenset(
    {"transparency_adapter", "robustness_adapter", "metrics_adapter", "reporter", "tracker"}
)
_VISUALISER_DECORATORS = frozenset({"transparency_visualiser", "robustness_visualiser"})


def _decorator_name(deco: ast.expr) -> str | None:
    """Return the bare name of a decorator call (e.g. ``metrics_adapter``)
    regardless of whether it was imported as ``metrics_adapter`` or accessed via
    ``module.metrics_adapter``."""
    if not isinstance(deco, ast.Call):
        return None
    func = deco.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


@lru_cache(maxsize=1)
def scan_adapter_extras() -> dict[str, str]:
    """Return ``{class_name: extra}`` harvested from raitap's source tree."""
    import raitap

    root = Path(raitap.__file__).resolve().parent
    found: dict[str, str] = {}
    for path in root.rglob("*.py"):
        # Tests can legitimately declare ``@<family>_adapter(..., extra="…")`` —
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
                name = _decorator_name(deco)
                if name not in _ADAPTER_DECORATORS and name not in _VISUALISER_DECORATORS:
                    continue
                assert isinstance(deco, ast.Call)  # narrowed by _decorator_name
                kwargs = {
                    kw.arg: kw.value.value
                    for kw in deco.keywords
                    if (
                        kw.arg is not None
                        and isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, str)
                    )
                }
                # ``extra`` defaults to ``registry_name`` at runtime for every
                # schema-backed adapter decorator; visualiser decorators don't
                # get an auto-extra — they ship with their parent adapter's extra.
                explicit_extra = kwargs.get("extra")
                if explicit_extra:
                    found[node.name] = explicit_extra
                elif name in _ADAPTER_DECORATORS:
                    registry_name = kwargs.get("registry_name")
                    if registry_name:
                        found[node.name] = registry_name
                break
    return found
