---
title: "Adding an adapter"
description: "Adapters (explainers, assessors, metrics, reporters, trackers) self-register via raitap.adapters.AdapterMixin. Adding a new one is one file plus optional pyproject.toml + test + docs entries. No central registry to edit."
myst:
  html_meta:
    "description": "Adapters (explainers, assessors, metrics, reporters, trackers) self-register via raitap.adapters.AdapterMixin. Adding a new one is one file plus optional pyproject.toml + test + docs entries. No central registry to edit."
---

# Adding an adapter

Adapters (explainers, assessors, metrics, reporters, trackers) self-register via
`raitap._adapters.AdapterMixin`. Adding a new one is **one file** plus
optional `pyproject.toml` + test + docs entries. No central registry to edit.

## The pattern

Concrete class declares its identity inline:

```python
class SuperXAIExplainer(
    AttributionOnlyExplainer,
    registry_name="superxai",        # CLI `+transparency=superxai` / Python `from raitap.transparency import superxai`
    extra="superxai",                 # uv extra mapped by raitap-deps
    library="superxai",               # pip package the adapter wraps
    error_patterns={                  # optional regex → friendly-message rewrite
        re.compile(r"some library footgun"): "Do X instead.",
    },
    suppress_warnings=(               # optional library-noise filters applied at import
        (r"some noisy.*pattern", UserWarning, r"superxai.*"),
    ),
):
    algorithm_registry = {"supertreeshap": frozenset({MethodFamily.SHAPLEY})}

    def __init__(self, algorithm: str, **init_kwargs): ...

    def _compute(self, model, inputs, **call_kwargs):
        superxai = self._lazy_import()       # no try/except boilerplate
        with self._rethrow():                # no rethrow(module=..., third_party_lib=..., ...) boilerplate
            return getattr(superxai, self.algorithm)(model, **self.init_kwargs).attribute(inputs, **call_kwargs)
```

`AdapterMixin.__init_subclass__` does everything else at module-load time:

1. Generates a hydra-zen builder typed against the family schema (`TransparencyConfig`).
2. Registers it in Hydra's `ConfigStore` under `(group="transparency", name="superxai")`.
3. Exposes it under `raitap.transparency.superxai` (lazy `__getattr__`).
4. Adds `("SuperXAIExplainer", "superxai")` to `ADAPTER_EXTRAS` for `raitap-deps`.
5. Adds `"superxai"` to `THIRD_PARTY_LIBS["transparency"]` so diagnostics attribute warnings/errors to it.
6. Applies any `suppress_warnings` filters once.
7. Wires `self._lazy_import()` (clean ImportError with install hint) and `self._rethrow()` (uses your `error_patterns`).

## Where each piece lives

| What | Where |
|---|---|
| Family group + schema (declared once) | The abstract base — e.g. `AttributionOnlyExplainer` declares `group="transparency"`, `schema=TransparencyConfig`. Concrete adapters inherit. |
| `registry_name` (default = snake-cased class name minus the family suffix) | Concrete class via `registry_name="..."`. |
| `extra` (optional dep) | Concrete class via `extra="..."`. |
| `library` (pip package name, wraps the third-party lib) | Concrete class via `library="..."`. |
| `error_patterns` (regex → friendlier message) | Concrete class via `error_patterns={re.compile(...): "..."}`. |
| `suppress_warnings` (library-noise filters) | Concrete class via `suppress_warnings=(("pattern", Category, "module_regex"),)`. |
| Optional Python package extras | `pyproject.toml` `[project.optional-dependencies] superxai = ["superxai-lib"]`. |

## Adapter families and where to look

| Module | Abstract base | Group | Schema |
|---|---|---|---|
| Transparency explainer | `raitap.transparency.explainers.base_explainer.AttributionOnlyExplainer` / `FullExplainer` | `transparency` | `TransparencyConfig` |
| Robustness assessor | `raitap.robustness.assessors.base_assessor.EmpiricalAttackAssessor` / `FormalVerificationAssessor` | `robustness` | `RobustnessConfig` |
| Metric | `raitap.metrics.base_metric.BaseMetricComputer` | `metrics` | `MetricsConfig` |
| Reporter | `raitap.reporting.base_reporter.BaseReporter` | `reporting` | `ReportingConfig` |
| Tracker | `raitap.tracking.base_tracker.BaseTracker` | `tracking` | `TrackingConfig` |
| Visualiser | `raitap.transparency.visualisers.base_visualiser.BaseVisualiser` / `raitap.robustness.visualisers.base_visualiser.BaseRobustnessVisualiser` | — (no Hydra group) | — |

## Adding a new algorithm to an existing adapter

No new class. Add a row to `algorithm_registry` on the existing adapter:

```python
class CaptumExplainer(AttributionOnlyExplainer, ...):
    algorithm_registry = {
        ...,
        "NewMethod": frozenset({MethodFamily.GRADIENT}),
    }
```

## Conventions

- **Class-keyword arguments**, not class-body assignments. Easy to omit when in the body; impossible to omit silently at the class declaration site (`AdapterMixin.__init_subclass__` raises `TypeError` if a concrete subclass has no inherited `group`/`schema`).
- **Abstract intermediates** (your own ABCs in the family) opt out with `abstract=True` on the class line. `inspect.isabstract(cls)` also auto-skips classes with unimplemented `@abstractmethod`s.
- **`package_style`** distinguishes dict-typed schema fields (`transparency: dict[str, TransparencyConfig]`, package=`"<group>.<name>"`) from flat schema fields (`metrics: MetricsConfig`, package=`"<group>"`). Set on the abstract base; concrete adapters inherit.

## When the mixin isn't enough

A handful of Hydra shapes can't be expressed via `AdapterMixin` alone and live as direct `ConfigStore` writes in `src/raitap/configs/zen.py::register_zen_groups`:

- **`# @package _global_` injections** — e.g. the `reporting=html` / `reporting=pdf` entries push both the `reporting` node **and** a `hydra.callbacks.reporting_sweep` block into the root config so multirun report aggregation auto-wires. The mixin always writes under `package="<group>"` or `"<group>.<name>"`; it can't reach `_global_`.
- **`_target_: null` variants** — e.g. `reporting=disabled` carries `_target_: null` + `multirun_report: false`. The mixin always targets a concrete class.
- **Anything that needs a custom hydra-zen `to_config=` or a non-dataclass node** — same escape hatch.

If your new adapter only needs to set `_target_` + optional kwargs on its schema (the 95% case), don't touch `zen.py`. Use class kwargs, done. If you genuinely need one of the special shapes above, add a `cs.store(group=..., name=..., package=..., node=...)` block in `register_zen_groups` after the `store.add_to_hydra_store(...)` flush — that order lets your specialisation overwrite the mixin-generated entry.

## End-to-end checklist for a new adapter

All four steps are **required**. None are optional.

1. **Adapter class.** Implement under `raitap/<module>/<subdir>/<name>_<entity>.py`. Inherit from the family abstract base (`AttributionOnlyExplainer`, `EmpiricalAttackAssessor`, `BaseMetricComputer`, …) and declare `registry_name`, `extra`, `library` (+ optional `error_patterns`, `suppress_warnings`) via class kwargs.

2. **`pyproject.toml` entry — mandatory.** Add `[project.optional-dependencies] <extra> = ["<pip-name>>=<min>"]`. The `raitap-deps` inference uses this to map `<extra>` (the class's `extra=` kwarg) → the install command emitted when a user composes a config that needs your adapter. Skipping this breaks auto-install for every user who composes your adapter — `raitap-deps` will refuse to emit a `uv sync` command because the extra doesn't exist. If your adapter belongs to an umbrella extra (e.g. `transparency = ["raitap[shap,captum,…]"]`), add yourself to that list too.

3. **Tests — mandatory.** Add `raitap/<module>/.../tests/test_<name>_<entity>.py`. Cover at minimum: registry membership, `check_backend_compat`, `__init_subclass__` wiring (the class lands in `_BUILDERS["<group>"]["<name>"]` and in `ADAPTER_EXTRAS`), and one `_compute` / `generate_adversarial` / `compute` happy path. CI enforces coverage on the module.

4. **Docs — mandatory.** Add a row to `docs/modules/<module>/frameworks-and-libraries.md` documenting the algorithms you support (or analogous page for non-adapter modules). YAML + Python tabs both required via `config-tabs`. This is what users see when they look up "does raitap support X?".

The framework only validates step 1 at class-creation time (`TypeError` if `group`/`schema` aren't inherited). Steps 2–4 break the user-facing contract silently; missing them blocks the PR in code review.
