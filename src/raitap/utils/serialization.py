from __future__ import annotations

from typing import Any


def to_json_serialisable(value: Any) -> Any:
    """Best-effort conversion to JSON-serialisable structures."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {str(k): to_json_serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_serialisable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return to_json_serialisable(value.item())
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass
    return repr(value)
