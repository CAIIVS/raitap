---
title: "Writing a plugin"
description: "Ship a RAITAP adapter as a standalone pip package — no fork, no PR required."
myst:
  html_meta:
    "description": "Ship a RAITAP adapter as a standalone pip package — no fork, no PR required."
---

# Writing a plugin

This page explains how to write a lightwight plugin adapter so your library can seamlessly be used via RAITAP. That way, you do not need to open a PR in the RAITAP repo, and consumers can use your library like any 1st party RAITAP adapter.

In the following guide, we will imagine you want to create 

## 1. Create the package

A plugin is an ordinary pip package. Lay it out like any `src/`-style project:

```
raitap-superxai/
├── pyproject.toml
└── src/
    └── raitap_superxai/
        └── __init__.py   # holds the decorated adapter; runs on import
```

## 2. Write the adapter

Implement the adapter exactly as in {doc}`adding-an-adapter` — the only
difference is your class lives in your own package, so import the base class by
its full path (`from raitap.transparency.explainers.base_explainer import ...`)
instead of the in-tree relative import. Decorate it with the public
`@adapters.<family>(...)` surface (`from raitap import adapters`). Same
`superxai-lib` transparency example as the adapter guide:

```python
# src/raitap_superxai/__init__.py
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from raitap import adapters
from raitap.transparency.contracts import MethodFamily
from raitap.transparency.explainers.base_explainer import AttributionOnlyExplainer

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


@adapters.transparency(
    registry_name="superxai",      # CLI `+transparency=superxai` / Python `from raitap.transparency import superxai`
    library="superxai-lib",        # real PyPI package name; drives `self._lazy_import()`
    error_patterns={               # rewrite cryptic upstream errors at call sites
        re.compile(r"some library footgun"): "Do X instead.",
    },
    algorithm_registry={
        "supertreeshap": frozenset({MethodFamily.SHAPLEY}),
    },
    onnx_compatible_algorithms=frozenset({"supertreeshap"}),
)
class SuperXAIExplainer(AttributionOnlyExplainer):
    def __init__(self, algorithm: str, **init_kwargs):
        super().__init__()
        self.algorithm = algorithm
        self.init_kwargs = init_kwargs

    def compute_attributions(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        backend=None,
        **call_kwargs,
    ) -> torch.Tensor:
        superxai = self._lazy_import()
        with self._rethrow():
            return getattr(superxai, self.algorithm)(model, **self.init_kwargs).attribute(
                inputs, **call_kwargs
            )
```

Decorator kwargs (`library`, `algorithm_registry`, `error_patterns`,
`suppress_warnings`, …) are documented in {doc}`adding-an-adapter`.
`AdapterDecoratorOptions` is exported for typing: `from raitap import
AdapterDecoratorOptions`.

## Step 3 — Declare the entry point and version pin

Two things in `pyproject.toml`: the `raitap.adapters` entry point (so RAITAP
finds your module) and a `raitap` dependency pin (so RAITAP can version-check
you).

```toml
[project]
name = "raitap-superxai"
dependencies = [
    "raitap>=0.5,<0.6",   # required — RAITAP reads this pin at load time
    "superxai-lib",
]

[project.entry-points."raitap.adapters"]
superxai = "raitap_superxai"   # value is the module to import; decorator fires on import
```

RAITAP reads the `Requires-Dist: raitap ...` metadata from your installed
distribution at load time. If the running RAITAP version doesn't satisfy the
pin — or if no `raitap` pin is declared at all — your plugin is **skipped with a
warning** and never breaks the user's run. Pin a tight range (`>=x,<y`) whenever
you rely on internal API that may change.

## Step 4 — Install and verify

Install your plugin alongside RAITAP and confirm it resolves like a first-party
adapter:

```bash
pip install raitap raitap-superxai
```

```bash
# resolves only if discovery + version check passed
python -c "from raitap.transparency import superxai; print(superxai)"
```

If nothing resolves, check the logs for a skip/crash warning naming your plugin
(see *How discovery works* below), or run with `RAITAP_DISABLE_PLUGINS` unset.

## Step 5 — Use it

Consumers reference your adapter by its `registry_name`, in YAML:

```yaml
transparency:
  my_run:
    _target_: SuperXAIExplainer
    algorithm: supertreeshap
```

or in Python:

```python
from raitap.transparency import superxai

transparency = {"my_run": superxai(algorithm="supertreeshap")}
```

## How discovery works

- Discovery fires at **config-registration time** (`register_zen_groups` /
  `register_configs`), not on a bare `import raitap`.
- Loading is **default-allow**: every installed plugin under the
  `raitap.adapters` entry-point group is discovered automatically.
- A plugin that **crashes at import** is logged (naming the plugin) and skipped
  — one bad plugin never breaks RAITAP.
- Set `RAITAP_DISABLE_PLUGINS=1` to skip all plugin discovery entirely.
