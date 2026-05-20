---
title: "Writing a plugin"
description: "Ship a RAITAP adapter as a standalone pip package — no fork, no PR required."
myst:
  html_meta:
    "description": "Ship a RAITAP adapter as a standalone pip package — no fork, no PR required."
---

# Writing a plugin

A **plugin** is a separate pip package that registers a RAITAP adapter via the
`raitap.adapters` entry-point group. No fork or pull request needed — once
installed alongside RAITAP the adapter appears in every family namespace just
like an in-tree one.

## Package layout

```
raitap-myattack/
├── pyproject.toml
└── src/
    └── raitap_myattack/
        └── __init__.py   # decorator fires on import
```

## The adapter

Implement the adapter exactly as described in {doc}`adding-an-adapter`. The only
difference is that your file lives in your own package instead of under
`src/raitap/`. The example below uses the robustness family:

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
`AdapterDecoratorOptions` is also exported for typing: `from raitap import
AdapterDecoratorOptions`.

## pyproject.toml

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

**Version pin.** RAITAP reads the `Requires-Dist: raitap ...` metadata from
your installed distribution at load time. If the running RAITAP version does not
satisfy the pin, or if no `raitap` pin is declared at all, the plugin is
**skipped with a warning** and never breaks the user's run. Pin a tight range
(`>=x,<y`) whenever you rely on internal API that may change.

## Discovery and isolation

- Discovery fires at **config-registration time** (`register_zen_groups` /
  `register_configs`), not on a bare `import raitap`.
- Loading is **default-allow**: all installed plugins under `raitap.adapters`
  are discovered automatically.
- A plugin that **crashes at import** is logged (naming the plugin) and skipped
  — it never breaks RAITAP.
- Set `RAITAP_DISABLE_PLUGINS=1` to skip all plugin discovery entirely.

## User consumption

```bash
pip install raitap raitap-myattack
```

Then in YAML:

```yaml
robustness:
  my_run:
    _target_: MyAttackAssessor
    algorithm: MyPGD
    constructor:
      eps: 0.03
```

Or in Python:

```python
from raitap.robustness import myattack

robustness = {"my_run": myattack(algorithm="MyPGD", constructor={"eps": 0.03})}
```
