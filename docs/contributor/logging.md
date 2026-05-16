---
title: "Logging and warnings"
description: "raitap exposes a single facade — raitap_log — for everything that would otherwise reach for logging.getLogger(_name_), warnings.warn, or a logger = logging.getLogger(...) boilerplate line. Use it."
myst:
  html_meta:
    "description": "raitap exposes a single facade — raitaplog — for everything that would otherwise reach for logging.getLogger(name), warnings.warn, or a logger = logging.getLogger(...) boilerplate line. Use it. Don't import logging or warnings in feature co"
---

# Logging and warnings

raitap exposes a single facade — `raitap_log` — for everything that would
otherwise reach for `logging.getLogger(__name__)`, `warnings.warn`, or a
`logger = logging.getLogger(...)` boilerplate line. **Use it.** Don't import
`logging` or `warnings` in feature code.

```python
from raitap import raitap_log

raitap_log.info("Running model forward pass...")
raitap_log.warn("Labels file is empty; falling back to predictions as targets.")
raitap_log.debug("Resolved %d sample ids", len(sample_ids))
raitap_log.exception("Tracker instantiation failed for target %r", target_path)
```

The facade lives at `src/raitap/utils/log.py` and is re-exported from the
top-level `raitap` package, so `from raitap import raitap_log` is the canonical
import.

## Why a facade?

Three reasons:

1. **One DX, no thinking.** Without it, every contributor has to choose
   between `warnings.warn`, `logger.info`, and `logger.warning` — and stamp out
   `logger = logging.getLogger(__name__)` in every file. With the facade,
   pick the verb (`.info`, `.debug`, `.warn`, `.error`, `.exception`,
   `.critical`) and ship.
2. **Caller-aware logger resolution.** `.info` / `.debug` / `.error` /
   `.exception` / `.critical` route to `logging.getLogger(<caller's __name__>)`
   via stack inspection. Per-module log levels configured through
   `logging.dictConfig` keep working — no `logger = …` boilerplate at the
   call site.
3. **Module-aware warnings.** `.warn` flows through `warnings.warn`, so it
   stays compatible with `warnings.filterwarnings`, `pytest.warns`, and
   `warnings.catch_warnings` — the rich console handler renders it as a
   framed panel with a *Module* chip and a *View docs* link, and
   `logging.captureWarnings` (installed by `setup_logging`) forwards it to
   the logging system so MLflow / Airflow / any handler attached to the root
   logger picks it up too.

## Methods

| Method                                              | Pipeline                                                 | Use for                                                                                                                                                                                                                                   |
| --------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `raitap_log.warn(msg, *args, *, module=…)`          | `warnings.warn` (also forwarded to `py.warnings` logger) | **Any warning.** User-facing config issues *and* operational signals (MLflow not ready, GPU fallback). Suppressible via `warnings.filterwarnings`, panel-rendered with module chip, captured by external log sinks via `captureWarnings`. |
| `raitap_log.info(msg, *args)`                       | `logging`                                                | Normal operational events (phase started, file written)                                                                                                                                                                                   |
| `raitap_log.debug(msg, *args)`                      | `logging`                                                | Verbose diagnostics off by default                                                                                                                                                                                                        |
| `raitap_log.error(msg, *args)`                      | `logging`                                                | Recoverable errors logged before fallback. Does **not** raise.                                                                                                                                                                            |
| `raitap_log.exception(msg, *args)`                  | `logging`                                                | Inside an `except` block — logs traceback at ERROR level                                                                                                                                                                                  |
| `raitap_log.critical(msg, *args)`                   | `logging`                                                | Process-ending failures                                                                                                                                                                                                                   |
| `raitap_log.suppress(message=, category=, module=)` | `warnings.filterwarnings`                                | Silence known-noise warnings from a wrapped library at adapter import time                                                                                                                                                                |

## One warning verb

There is exactly one warning verb: **`.warn`**. Both user-config issues
("labels file is empty") and operational signals ("MLflow server not ready
after 10s") use it. The rich handler frames every warning as a panel with a
module chip; `logging.captureWarnings` forwards every warning to the
logging system so MLflow / Airflow / external sinks see them.

If you find yourself wanting an "operational warning that doesn't show as a
panel," ask whether the call site is really:

- **A failure?** Use `.error` (or `.exception` if you're inside an `except`
  block) — these are not warnings, they're errors logged before recovery.
- **Routine progress?** Use `.info`.

`.warn` is the right call for anything in between — including the cases that
used to call `logger.warning(..., exc_info=True)`. If you have a traceback to
attach, you almost always want `.exception` (ERROR level) rather than a
warning anyway.

## Raising errors

Don't do it through `raitap_log`. Use plain `raise` for raitap-originated
errors, and use the `rethrow` context manager when wrapping a third-party
library that throws confusing messages:

```python
raise ValueError(
    f"Data source {source!r} does not exist.\n"
    f"Expected a URL, an existing local path, or a named demo sample."
)
```

For wrapped third-party calls (captum / shap / foolbox / torchattacks), use
`raitap.utils.errors.rethrow` to rewrap matched error messages into a
user-actionable `AdapterError` carrying a `Diagnostic`. The rich handler
renders raised `RaitapError` subclasses with the same `Module · via <lib>
· View docs` chips as warnings, and the top-level `print_failure_panel`
suppresses the raw traceback so the user sees the actionable copy first.

```python
# src/raitap/transparency/explainers/shap_explainer.py
import re
from raitap.utils.diagnostics import Module
from raitap.utils.errors import rethrow

class ShapExplainer:
    # Curated patterns: original message regex → replacement copy.
    # Matched against ``str(exc)``; first hit wins. Unmatched errors propagate
    # unchanged so real bugs don't get masked.
    error_messages = {
        re.compile(r"BackwardHookFunctionBackward is a view"): (
            "DeepExplainer can fail on PyTorch models that use SiLU "
            "activations (for example EfficientNet variants). Use "
            "alternatives like GradientExplainer."
        ),
    }

    def compute_attributions(self, model, inputs, ...):
        ...
        with rethrow(
            module=Module.transparency,
            third_party_lib="shap",
            message_map=type(self).error_messages,
        ):
            shap_values = explainer.shap_values(inputs)
```

The original exception is preserved on `__cause__`, so the raw library
traceback stays available for debugging — only the user-facing top is
rewritten.

## Module attribution

The rich handler decorates warning panels with a module chip (Metrics,
Robustness, Transparency, …) and a *View docs* link. It infers the module
from the call site by walking frames at `warnings.formatwarning` time —
usually correct because the call site lives inside the right module
directory.

When the **logical** module differs from the file path, pass it explicitly:

```python
# src/raitap/pipeline/pipeline.py — file lives in pipeline/, but the warning
# is logically a robustness concern.
from raitap import raitap_log
from raitap.utils.diagnostics import Module

raitap_log.warn(
    "No ground-truth labels provided; using model predictions as the "
    "reference for untargeted attacks.",
    module=Module.robustness,
)
```

`Module` is a `StrEnum` with one member per top-level raitap directory.
Use the enum, not a raw string, so a typo fails at import.

## Suppressing third-party noise

Adapters that wrap a noisy library should call `raitap_log.suppress` at
**module import time** so the filter is in place before the first call:

```python
# src/raitap/transparency/explainers/captum_explainer.py
from raitap import raitap_log

# Captum emits this on every run when inputs don't already require gradients.
# It auto-fixes the issue, so the warning is pure noise — silence it at import.
# Scope ``module=`` to captum so unrelated UserWarnings whose messages happen
# to match the same pattern aren't accidentally hidden.
raitap_log.suppress(
    message=r"Input Tensor.*required_grads",
    category=UserWarning,
    module=r"captum.*",
)
```

Always scope `module=` to the wrapped library so unrelated UserWarnings with
the same message text don't get hidden.

## What you should never write

```python
# ❌ Don't do this — facade exists for a reason.
import logging
logger = logging.getLogger(__name__)

# ❌ Don't do this either.
import warnings
warnings.warn("…", UserWarning, stacklevel=2)
```

Both patterns lose module attribution and force every reader to track two
mental models. Use `raitap_log`.

## Where the infrastructure lives

- `src/raitap/utils/log.py` — the `_RaitapLog` class and `raitap_log`
  singleton. Contains the thread-local diagnostic queue that bridges
  `warnings.formatwarning` (frames at warn time) to the rich handler
  (panels at emit time).
- `src/raitap/utils/diagnostics.py` — `Module` enum, frame-walking
  classifier, third-party library detection (libs declared in each
  module's `__init__.py` as `THIRD_PARTY_LIBS`).
- `src/raitap/utils/errors.py` — `RaitapError`, `AdapterError`,
  traceback-walking diagnostic resolver, and the `rethrow` context manager
  that adapters use to rewrap confusing third-party errors.
- `src/raitap/utils/colour.py` — two-shade colour palette (`<hue>_base` /
  `<hue>_light`) plus the Rich `Theme`. Edit here when adding or
  rebalancing colours; renderers reference tokens, not raw ANSI names.
- `src/raitap/utils/console.py` — the rich `RichHandler` subclass that
  formats WARNING+ records into panels with module / via-lib / docs
  chips. Calls `logging.captureWarnings(True)` so external log sinks
  (MLflow, Airflow) see warnings too. `print_failure_panel` mirrors the
  same chip composition for top-level crashes.

If you're touching that code, expect to also update
`src/raitap/utils/tests/test_log.py`,
`src/raitap/utils/tests/test_diagnostics.py`,
`src/raitap/utils/tests/test_errors.py`, and
`src/raitap/utils/tests/test_console_errors.py`.
