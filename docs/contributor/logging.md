---
title: "Logging and warnings"
description: "raitap_log is the single facade for logging + warnings. Don't import logging or warnings in feature code — use raitap_log."
myst:
  html_meta:
    "description": "raitap_log is the single facade for logging + warnings. Don't import logging or warnings in feature code — use raitap_log."
---

# Logging and warnings

`raitap_log` is the single facade for `logging.getLogger(__name__)` + `warnings.warn`. **Don't import `logging` or `warnings` in feature code.**

```python
from raitap import raitap_log

raitap_log.info("Running model forward pass...")
raitap_log.warn("Labels file is empty; falling back to predictions as targets.")
raitap_log.debug("Resolved %d sample ids", len(sample_ids))
raitap_log.exception("Tracker instantiation failed for target %r", target_path)
```

Lives at `src/raitap/utils/log.py`, re-exported as `raitap.raitap_log`.

## Methods

| Method | Pipeline | Use for |
|---|---|---|
| `.warn(msg, *args, *, module=…)` | `warnings.warn` (forwarded to `py.warnings` logger) | Any warning. User-config issues + operational signals. Panel-rendered with module chip, suppressible via `warnings.filterwarnings`, captured by external sinks. |
| `.info(msg, *args)` | `logging` | Normal operational events (phase started, file written). |
| `.debug(msg, *args)` | `logging` | Verbose diagnostics off by default. |
| `.error(msg, *args)` | `logging` | Recoverable errors logged before fallback. Does **not** raise. |
| `.exception(msg, *args)` | `logging` | Inside `except` — logs traceback at ERROR. |
| `.critical(msg, *args)` | `logging` | Process-ending failures. |
| `.suppress(message=, category=, module=)` | `warnings.filterwarnings` | Silence known-noise library warnings. Prefer the `suppress_warnings=` decorator kwarg on adapters (see below). |

Caller-aware: `.info` / `.debug` / `.error` / `.exception` / `.critical` route to `logging.getLogger(<caller's __name__>)` via stack inspection — no `logger = …` boilerplate needed.

## One warning verb

Exactly one: **`.warn`**. Covers user-config issues AND operational signals.

If you want something more dramatic, ask:
- **Failure?** → `.error` (or `.exception` inside `except`).
- **Routine progress?** → `.info`.

`.warn` covers everything in between. If you have a traceback, you almost always want `.exception` instead.

## Raising errors

Don't go through `raitap_log` — use plain `raise` for raitap-originated errors:

```python
raise ValueError(
    f"Data source {source!r} does not exist.\n"
    "Expected a URL, an existing local path, or a named demo sample."
)
```

For wrapped third-party calls (captum / shap / foolbox / torchattacks), use the adapter's `self._rethrow()` helper. It pulls `library`, the family group, and the `error_patterns` map straight from the adapter's `@adapters.<family>(...)` decoration — no kwargs needed at the call site:

```python
# src/raitap/transparency/explainers/shap_explainer.py
from raitap import adapters

@adapters.transparency(
    registry_name="shap",
    library="shap",
    error_patterns={
        r"BackwardHookFunctionBackward is a view": (
            "DeepExplainer can fail on PyTorch models that use SiLU "
            "activations (for example EfficientNet variants). Use "
            "alternatives like GradientExplainer."
        ),
    },
    algorithm_registry={...},
)
class ShapExplainer(AttributionOnlyExplainer):
    def compute_attributions(self, model, inputs, **kwargs):
        shap = self._lazy_import()
        with self._rethrow():
            return getattr(shap, self.algorithm)(model, ...).shap_values(inputs)
```

The original exception is preserved on `__cause__`, so the raw traceback stays for debugging — only the user-facing top is rewritten. `RaitapError` subclasses render with the same `Module · via <lib> · View docs` chips as warnings.

## Module attribution

The rich handler decorates warnings with a module chip (Metrics, Robustness, Transparency, …). Inferred from the call site via frame walking — usually correct.

When the **logical** module differs from the file path, pass `module=` explicitly:

```python
# src/raitap/pipeline/phases/assess_robustness.py — pipeline file,
# logically a robustness concern.
raitap_log.warn(
    "No ground-truth labels provided; using model predictions as the "
    "reference for untargeted attacks.",
    module=Module.robustness,
)
```

`Module` is a `StrEnum` (one member per top-level raitap directory). Use the enum, not a raw string.

## Suppressing third-party noise

Use the `suppress_warnings=` decorator kwarg on the adapter, not a module-level `raitap_log.suppress` call. The decorator installs the filter at registration time (same as a module-level call would), but the noise spec stays colocated with the adapter that owns it:

```python
from raitap import adapters

@adapters.transparency(
    registry_name="captum",
    library="captum",
    suppress_warnings=[
        # Captum emits this on every run when inputs don't already require
        # gradients. Auto-fixes the issue → pure noise. Scope module=captum
        # so unrelated UserWarnings with matching text aren't hidden.
        (r"Input Tensor.*required_grads", UserWarning, r"captum.*"),
    ],
    algorithm_registry={...},
)
class CaptumExplainer(AttributionOnlyExplainer): ...
```

Always scope `module=` to the wrapped library so unrelated UserWarnings with the same text aren't hidden.

## Where the infrastructure lives

- `src/raitap/utils/log.py` — `_RaitapLog` class + `raitap_log` singleton. Owns the thread-local diagnostic queue bridging `warnings.formatwarning` to the rich handler.
- `src/raitap/utils/diagnostics.py` — `Module` enum + frame-walking classifier + third-party library detection. The library set is auto-populated by `_register_core` from each `@register_*_adapter(..., library="...")` decoration and stored at `raitap._adapters.THIRD_PARTY_LIBS` (grouped by adapter family).
- `src/raitap/utils/errors.py` — `RaitapError` / `AdapterError`, traceback-walking diagnostic resolver, `rethrow` context manager.
- `src/raitap/utils/colour.py` — two-shade palette + Rich `Theme`. Edit here when adding/rebalancing colours.
- `src/raitap/utils/console.py` — `RichHandler` subclass + `print_failure_panel`. Calls `logging.captureWarnings(True)` so external sinks see warnings.

Touching that code? Expect to update `src/raitap/utils/tests/test_log.py`, `test_diagnostics.py`, `test_errors.py`, `test_console_errors.py`.
