from __future__ import annotations

from typing import Any


def ensure_algorithm_in_allowlist(
    selected_algorithm: str,
    allowlist: frozenset[str],
    *,
    error_cls: type[Exception],
    **error_kwargs: Any,
) -> None:
    """Validate that ``algorithm`` is allowed when a non-empty allowlist is configured."""
    if not allowlist:
        return
    if selected_algorithm in allowlist:
        return
    raise error_cls(**error_kwargs)
