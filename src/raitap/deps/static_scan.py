"""Best-effort static scan of adapter ``extra=`` declarations.

Walks the installed raitap source tree with :mod:`ast` and harvests every
``class Foo(..., extra="bar"):`` declaration without importing the module
(and therefore without pulling the wrapped third-party library).

Used by :func:`raitap.deps.inference._extra_for_target` so the deps
bootstrap can resolve ``_target_`` → extra mappings in partial-extras
venvs — i.e. before the very libraries it is about to install have been
installed. The runtime ``ADAPTER_EXTRAS`` dict populated by
:meth:`raitap._adapters.AdapterMixin.__init_subclass__` is still the
authoritative source whenever it is populated; this scanner is a safety
net for ``raitap.run(..., auto_install=True)`` and the CLI bootstrap.

Preserves the "adding a new adapter = single-file add" invariant:
nothing here is hand-maintained, and a new adapter's ``extra="..."``
kwarg is picked up automatically.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def scan_adapter_extras() -> dict[str, str]:
    """Return ``{class_name: extra}`` harvested from raitap's source tree."""
    import raitap

    root = Path(raitap.__file__).resolve().parent
    found: dict[str, str] = {}
    for path in root.rglob("*.py"):
        # Tests can legitimately declare ``class Fake(..., extra="…")`` —
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
            for kw in node.keywords:
                if (
                    kw.arg == "extra"
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)
                ):
                    found[node.name] = kw.value.value
                    break
    return found
