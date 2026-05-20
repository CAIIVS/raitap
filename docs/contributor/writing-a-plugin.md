---
title: "Writing a plugin"
description: "Ship a RAITAP adapter as a standalone pip package — no fork, no PR required."
myst:
  html_meta:
    "description": "Ship a RAITAP adapter as a standalone pip package — no fork, no PR required."
---

# Writing a plugin

This page explains how to write a lightwight plugin adapter so your library can seamlessly be used via RAITAP. That way, you do not need to open a PR in the RAITAP repo, and consumers can use your library like any 1st party RAITAP adapter.


## Step 1 — Create the package

A plugin is an ordinary pip package. Lay it out like any `src/`-style project:

```
raitap-myattack/
├── pyproject.toml
└── src/
    └── raitap_myattack/
        └── __init__.py   # holds the decorated adapter; runs on import
```

## Step 2 — Write the adapter

Implement the adapter exactly as in {doc}`adding-an-adapter` — the only
difference is your class lives in your own package, not under `src/raitap/`.
Decorate it with the public `@adapters.<family>(...)` surface (`from raitap
import adapters`). Example, robustness family:

```python
# src/raitap_myattack/__init__.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap import adapters
from raitap.robustness.assessors.base_assessor import EmpiricalAttackAssessor
from raitap.robustness.contracts import MethodKind, Objective, PerturbationNorm, ThreatModel
from raitap.robustness.semantics import AssessorSemanticsHints
from raitap.utils.lazy import lazy_import

if TYPE_CHECKING:
    import torch
    from torch import nn
else:
    torch = lazy_import("torch")


@adapters.robustness(
    registry_name="myattack",           # CLI `+robustness=myattack` / Python `from raitap.robustness import myattack`
    library="myattack-lib",             # real PyPI name; drives self._lazy_import()
    algorithm_registry={
        "MyPGD": AssessorSemanticsHints(
            MethodKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,
            families=frozenset({"gradient_sign", "iterative"}),
        ),
    },
)
class MyAttackAssessor(EmpiricalAttackAssessor):
    def __init__(self, algorithm: str, **init_kwargs: Any) -> None:
        self.algorithm = algorithm
        self.init_kwargs = dict(init_kwargs)

    def generate_adversarial(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        backend: object | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        lib = self._lazy_import()
        with self._rethrow():
            attack = getattr(lib, self.algorithm)(model, **self.init_kwargs)
            return attack(inputs, targets)
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
name = "raitap-myattack"
dependencies = [
    "raitap>=0.5,<0.6",   # required — RAITAP reads this pin at load time
    "myattack-lib",
]

[project.entry-points."raitap.adapters"]
myattack = "raitap_myattack"   # value is the module to import; decorator fires on import
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
pip install raitap raitap-myattack
```

```bash
# resolves only if discovery + version check passed
python -c "from raitap.robustness import myattack; print(myattack)"
```

If nothing resolves, check the logs for a skip/crash warning naming your plugin
(see *How discovery works* below), or run with `RAITAP_DISABLE_PLUGINS` unset.

## Step 5 — Use it

Consumers reference your adapter by its `registry_name`, in YAML:

```yaml
robustness:
  my_run:
    _target_: MyAttackAssessor
    algorithm: MyPGD
    constructor:
      eps: 0.03
```

or in Python:

```python
from raitap.robustness import myattack

robustness = {"my_run": myattack(algorithm="MyPGD", constructor={"eps": 0.03})}
```

## How discovery works

- Discovery fires at **config-registration time** (`register_zen_groups` /
  `register_configs`), not on a bare `import raitap`.
- Loading is **default-allow**: every installed plugin under the
  `raitap.adapters` entry-point group is discovered automatically.
- A plugin that **crashes at import** is logged (naming the plugin) and skipped
  — one bad plugin never breaks RAITAP.
- Set `RAITAP_DISABLE_PLUGINS=1` to skip all plugin discovery entirely.
