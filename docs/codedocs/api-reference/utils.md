---
title: "Utils API"
description: "Reference for the small public utility surface exported by raitap.utils."
---

The public utilities surface is intentionally tiny. `raitap.utils` currently re-exports one helper that the rest of the package uses for metadata and artifact serialization.

## Import

```python
from raitap.utils import to_json_serialisable
```

## `to_json_serialisable`

Source: `src/raitap/utils/serialization.py`

```python
def to_json_serialisable(value: Any) -> Any
```

Best-effort conversion of nested Python values into JSON-safe structures.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `value` | `Any` | — | Arbitrary object or nested structure. |

Return type: `Any`

Behavior:

- primitive JSON values are returned unchanged
- dict keys are stringified recursively
- lists, tuples, and sets become lists
- scalar tensor-like objects that implement `.item()` are reduced to Python values
- unsupported objects fall back to `repr(value)`

Example:

```python
from raitap.utils import to_json_serialisable

payload = {
    "score": 0.98,
    "classes": {1, 2, 3},
}

print(to_json_serialisable(payload))
```

This helper is used by metrics, reporting, and transparency metadata writers. It is small, but it is one of the reasons artifacts from different modules stay structurally consistent.

## Typical usage in RAITAP internals

You will see `to_json_serialisable()` in three recurring situations:

- metrics persistence in `src/raitap/metrics/factory.py`
- transparency metadata conversion in `src/raitap/transparency/results.py`
- report manifest serialization in `src/raitap/reporting/manifest.py`

That reuse is important because each subsystem emits slightly different Python objects. Metrics may contain tensors converted through `.item()`, transparency metadata may contain enums and nested dataclasses, and report structures may contain paths or tuples. `to_json_serialisable()` does not solve every case on its own, but it provides the common last-mile coercion that keeps those artifacts JSON-friendly.

If you are extending RAITAP with a custom tracker, visualiser, or reporter, this is the public helper to reach for before writing your own ad hoc conversion code.

It is intentionally conservative: if a value cannot be converted safely, the helper prefers a readable representation over a lossy silent cast.
