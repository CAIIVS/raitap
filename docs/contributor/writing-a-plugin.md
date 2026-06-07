---
title: "Writing a plugin"
description: "Ship a RAITAP adapter as a standalone pip package. No fork, no PR required."
myst:
  html_meta:
    "description": "Ship a RAITAP adapter as a standalone pip package. No fork, no PR required."
---

# Writing a plugin

This page explains how to write a lightweight plugin adapter so your library can seamlessly be used
via RAITAP. That way, you do not need to open a PR in the RAITAP repo, and consumers can use your
library like any 1st party RAITAP adapter.

In the following guide, we will imagine you want to make your "SuperXAI" library usable to RAITAP
users seamlessly.

## Supported modules

Plugins can register:

- **Transparency** explainers & visualisers
- **Robustness** assessors & visualisers
- **Metrics** computers
- **Reporting** renderers
- **Trackers**

Backends are not yet plugin-extensible. You can open a feature request or a PR for 1st party support.

## 1. Create the package

A plugin is an ordinary pip package. Lay it out like any `src/`-style project:

```
raitap-superxai/
├── pyproject.toml
└── src/
    └── raitap_superxai/
        └── __init__.py   # holds the decorated adapter; runs on import (see below)
```

## 2. Write the adapter

Implement the adapter exactly as in {doc}`adding/adding-an-adapter`. The only difference is your class
lives in your own package, so import the base class by its full path
(`from raitap.transparency.explainers.base_explainer import ...`) instead of the in-tree relative
import. Decorate it with the public `@adapters.<family>(...)` surface (`from raitap import adapters`).
Same `superxai-lib` transparency example as the adapter guide:

```python
# src/raitap_superxai/__init__.py

from __future__ import annotations

from typing import TYPE_CHECKING

from raitap import adapters
from raitap.transparency.contracts import ExplainerAlgorithmSpec, MethodFamily
from raitap.transparency.explainers.base_explainer import AttributionOnlyExplainer
from raitap.types import Capability

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


@adapters.transparency(
    registry_name="superxai",      # CLI `+transparency=superxai` / Python `from raitap.transparency import superxai`
    library="superxai-lib",        # real name of your PyPI package; drives `self._lazy_import()` (defaults to registry_name)
    error_patterns={               # rewrite cryptic upstream errors at call sites
        r"some library footgun": "Do X instead.", # nicer error messages to avoid deep stack traces in RAITAP
    },
    algorithm_registry={
        # the algos your library offers; ExplainerAlgorithmSpec carries the
        # method families (+ optional baseline_default for reference-input methods)
        # and the capability requirements for the algorithm.
        # empty requires (default) = model-agnostic, runs on ONNX/forward-only backends
        "supertreeshap": ExplainerAlgorithmSpec({MethodFamily.SHAPLEY}),
        # requires={Capability.AUTOGRAD} for algorithms that need autograd,
        # e.g. gradient-based methods:
        # "supergrad": ExplainerAlgorithmSpec({MethodFamily.GRADIENT},
        #                                       requires={Capability.AUTOGRAD}),
    },
)
class SuperXAIExplainer(AttributionOnlyExplainer):
    def __init__(self, algorithm: str, **init_kwargs):
        super().__init__()              # don't omit!
        self.algorithm = algorithm
        self.init_kwargs = init_kwargs

    def compute_attributions(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        backend=None,
        **call_kwargs,
    ) -> torch.Tensor:
        superxai = self._lazy_import()  # don't omit!
        with self._rethrow():
            return getattr(superxai, self.algorithm)(model, **self.init_kwargs).attribute(
                inputs, **call_kwargs
            )
```

Decorator kwargs (`library`, `algorithm_registry`, `error_patterns`,
`suppress_warnings`, ...) are documented in {doc}`adding/adding-an-adapter`.

`AdapterDecoratorOptions` is exported for typing, in case you want additional custom logic on top of
the decorator: `from raitap import AdapterDecoratorOptions`.

**Multiple modules.** One plugin can register adapters for several RAITAP modules. Add more
decorated classes to your `__init__.py` (e.g. an `@adapters.robustness(...)` assessor next to the
explainer). Every decorator fires when the plugin is imported on discovery.

**Custom rendering.** By default your plugin's image attributions render with raitap's built-in
style, nothing extra to declare. If you want to ship a custom look, register one renderer:

```python
from raitap import adapters

@adapters.image_renderer(for_library="superxai")  # your registry_name
class SuperXAIRenderer:
    def draw(self, ax, attr, image, *, sign="all", **style):
        ...  # paint attr onto ax; return the mappable (or None)
```

The renderer then applies automatically to both classification and detection attribution maps.

## 3. Declare the entry point and version pin

Two things to add to your `pyproject.toml`:

- the `raitap.adapters` entry point (so RAITAP finds your module)
- a `raitap` dependency pin (so RAITAP can version-check you).

```toml
[project]
name = "raitap-superxai"  # plugin name, not your published PyPI package
dependencies = [
    "raitap>=0.5,<0.6",   # required: RAITAP reads this pin at load time
    "superxai-lib",       # your published PyPI package
]

[project.entry-points."raitap.adapters"]
superxai = "raitap_superxai"   # name of the file in src (see Step 1), NOT YOUR PLUGIN NAME
```

RAITAP reads the `Requires-Dist: raitap ...` metadata from your installed distribution at load time.
If the running RAITAP version doesn't satisfy the pin, or if no `raitap` pin is declared at all,
your plugin is **skipped with a warning** and never breaks the user's run. Pin a tight range
(`>=x,<y`) whenever you rely on internal API that may change.

## 4. Do a self-test

Install your plugin alongside RAITAP and confirm it resolves like a first-party adapter:

```{install-tabs}
:uv:
uv add raitap raitap-superxai

:pip:
pip install raitap raitap-superxai
```

```bash
# resolves only if discovery + version check passed
python -c "from raitap.transparency import superxai; print(superxai)"
```

If nothing resolves, check the logs for a skip/crash warning naming your plugin
(see {ref}`disco`), or run with `RAITAP_DISABLE_PLUGINS` unset.

## 5. Document consumer usage

RAITAP already documents how to use plugins ({doc}`../using-raitap/using-plugins`), but it's always
good to add a section in your own docs too.

(disco)=

## How discovery works

- Discovery fires at **config-registration time** (`register_zen_groups` /
  `register_configs`), not on a bare `import raitap`.
- Loading is **default-allow**: every installed plugin under the
  `raitap.adapters` entry-point group is discovered automatically.
- A plugin that **crashes at import** is logged (naming the plugin) and skipped.
  One bad plugin never breaks RAITAP.
- Set `RAITAP_DISABLE_PLUGINS=1` to skip all plugin discovery entirely.
