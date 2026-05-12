from __future__ import annotations

import re


def bucket_class(bucket: object, correct: object = None) -> str:
    if correct is True:
        return "success"
    if correct is False:
        return "danger"
    value = "" if bucket is None else str(bucket).lower()
    if value in {"high_confidence", "correct", "success"}:
        return "success"
    if value in {"insecure", "uncertain", "warning"}:
        return "warn"
    if value in {"wrong", "incorrect", "danger"}:
        return "danger"
    return "neutral"


def asr_band(value: object) -> str:
    parsed = _to_float(value)
    if parsed is None:
        return "neutral"
    if parsed < 0.2:
        return "low"
    if parsed < 0.5:
        return "mid"
    return "high"


def fmt_num(value: object, n: int = 3) -> str:
    parsed = _to_float(value)
    if parsed is None:
        return "n/a" if value is None or str(value).strip() == "" else str(value)
    return f"{parsed:.{n}f}".rstrip("0").rstrip(".")


def fmt_pct(value: object) -> str:
    parsed = _to_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed * 100:.1f}%"


def as_dict(table_rows: tuple[tuple[str, str], ...]) -> dict[str, str]:
    return dict(table_rows)


def slug(value: object) -> str:
    text = "n-a" if value is None else str(value).strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "-", text)
    return text.strip("-") or "n-a"


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, str | int | float):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
