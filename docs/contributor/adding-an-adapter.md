# Adding an adapter

Adapters (explainers, assessors, metrics, reporters, trackers) self-register via
`raitap._adapters.AdapterMixin`. Adding a new one is **one file** plus
optional `pyproject.toml` + test + docs entries. No central registry to edit.

## The pattern

Concrete class declares its identity inline:

```python
class SuperXAIExplainer(
    AttributionOnlyExplainer,
    registry_name="superxai",   # CLI: `+transparency=superxai` / Python: `from raitap.transparency import superxai`
    extra="superxai",            # uv extra mapped by raitap-deps
):
    algorithm_registry = {"supertreeshap": frozenset({MethodFamily.SHAPLEY})}
    def __init__(self, algorithm: str, **init_kwargs): ...
    def _compute(self, model, inputs, **call_kwargs): ...
```

That triggers `__init_subclass__` once at module-load time:

1. Generates a hydra-zen builder typed against the family schema (`TransparencyConfig`).
2. Registers it in Hydra's `ConfigStore` under `(group="transparency", name="superxai")`.
3. Adds it to `_BUILDERS["transparency"]["superxai"]` so `from raitap.transparency import superxai` resolves.
4. Adds `("SuperXAIExplainer", "superxai")` to `ADAPTER_EXTRAS` so `raitap-deps` knows the optional dep.

## Where each piece lives

| What | Where |
|---|---|
| Family group + schema (declared once) | The abstract base — e.g. `AttributionOnlyExplainer` declares `group="transparency"`, `schema=TransparencyConfig`. Concrete adapters inherit. |
| `registry_name` (default = snake-cased class name minus the family suffix) | Concrete class via `registry_name="..."`. |
| `extra` (optional dep) | Concrete class via `extra="..."`. |
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

## End-to-end checklist for a new adapter

1. Implement the class in the right `raitap/<module>/<subdir>/` location.
2. Add `[project.optional-dependencies]` entry in `pyproject.toml` if it pulls a new lib.
3. Tests under `raitap/<module>/.../tests/test_<name>.py`.
4. Docs row in `docs/modules/<module>/frameworks-and-libraries.md` (or analogous).

Steps 2–4 are convention; only step 1 is enforced by the framework.
