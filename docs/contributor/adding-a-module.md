---
title: "Adding a module"
description: "How to add a new top-level RAITAP module (like transparency / robustness / metrics) — new family decorator, schema, pipeline phase, and base classes."
myst:
  html_meta:
    "description": "How to add a new top-level RAITAP module (like transparency / robustness / metrics) — new family decorator, schema, pipeline phase, and base classes."
---

# Adding a module

A **module** is a top-level RAITAP capability (transparency, robustness, metrics, reporting, tracking). Adding one is a much bigger change than adding an adapter — you create a new family decorator, a new schema, a new pipeline phase, and the base class everyone in the module subclasses.

**Before you start**: confirm the new capability genuinely doesn't fit an existing module. A new attack library is a new robustness adapter, not a new module. A new "fairness" capability — with its own result type, its own visualisations, its own report section — is a new module.

The walkthrough below uses a fictional `fairness` module.

## Files you will create or edit

| Path | What |
|---|---|
| `src/raitap/fairness/__init__.py` | Re-exports + lazy `__getattr__` for `from raitap.fairness import <adapter>` |
| `src/raitap/fairness/contracts.py` | Domain types (enums, dataclasses, `MethodFamily`-equivalent) |
| `src/raitap/fairness/results.py` | `FairnessResult` dataclass + `write_artifacts()` |
| `src/raitap/fairness/exceptions.py` | `FairnessBackendIncompatibilityError`, etc. |
| `src/raitap/fairness/assessors/__init__.py` | Subdir for the concrete adapters |
| `src/raitap/fairness/assessors/base_assessor.py` | Abstract base — subclasses implement one method |
| `src/raitap/fairness/assessors/registration.py` | `register_fairness_adapter` family decorator |
| `src/raitap/fairness/factory.py` | Iterates `config.fairness` dict, instantiates + runs each adapter |
| `src/raitap/fairness/report.py` | `FairnessPhase(AssessmentPhase)` + `FairnessPhaseResult(Trackable, Reportable)` — your phase (check + run) and its result (logging + report sections), co-located |
| `src/raitap/configs/schema.py` | Add `FairnessConfig` + `fairness:` field on `AppConfig` |
| `src/raitap/pipeline/phases/assess_fairness.py` | The phase *work* function the `FairnessPhase.run` calls (mirrors `assess_transparency`) |
| `src/raitap/pipeline/phases/registry.py` | Import `FairnessPhase` + add one entry to `ASSESSMENT_PHASES` — the only pipeline edit |
| `pyproject.toml` | Optional: `fairness = [...]` extra if the module wraps libraries |
| `docs/modules/fairness/*.md` | User-facing docs + `frameworks-and-libraries.md` |
| `tests/...` | Per-adapter tests + family E2E + `test_partial_extras_safe.py` |

## 1. Domain contracts (`contracts.py`)

Mirror the shape of `src/raitap/transparency/contracts.py` or `src/raitap/robustness/contracts.py`. Declare:

- Enums for the kinds your module distinguishes (e.g. `FairnessMetricKind`, `ProtectedAttributeKind`).
- The `Mapping[str, T]` value type that goes into `algorithm_registry` (transparency uses `ExplainerSemanticsHints`; robustness uses `AssessorSemanticsHints` — both are frozen per-algorithm hints dataclasses).
- A `Protocol` for the adapter (optional but useful for cross-module callers).

## 2. Base class (`assessors/base_assessor.py`)

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from raitap._adapters import AdapterMixin

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..results import FairnessResult
    from .semantics import FairnessAdapterSemanticsHints


class BaseFairnessAssessor(AdapterMixin, ABC):
    """Root base for every fairness adapter.

    Concrete subclasses must declare ``algorithm_registry`` as a class-body
    ClassVar — enforced at decoration time by ``@register_fairness_adapter``.
    """

    algorithm_registry: ClassVar[Mapping[str, FairnessAdapterSemanticsHints]]
    # Adapter-specific; defaults to "no ONNX support". The decorator overrides per-adapter.
    ONNX_COMPATIBLE_ALGORITHMS: ClassVar[frozenset[str]] = frozenset()

    @abstractmethod
    def assess(self, model, inputs, sensitive_attrs, **kwargs) -> FairnessResult:
        """Produce a fairness assessment."""

    def check_backend_compat(self, backend: object) -> None:
        """Default: autograd OR allowlisted ONNX algorithm. Override only if
        your contract differs."""
        # See raitap/robustness/assessors/base_assessor.py for the canonical
        # implementation — copy-paste, swap the error class for your module's.
        ...
```

The `check_backend_compat` default + `ONNX_COMPATIBLE_ALGORITHMS` plumbing is identical across modules; mirror what transparency / robustness already do.

## 3. Family decorator (`assessors/registration.py`)

```python
from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Unpack

from raitap._adapters import (
    ALL,
    FamilyConfig,
    _AllAlgorithmsSentinel,
    AdapterDecoratorOptions,
    _register_core,
)
from raitap.configs.schema import FairnessConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from raitap.fairness.contracts import FairnessAdapterSemanticsHints
    from raitap.fairness.assessors.base_assessor import BaseFairnessAssessor

FAIRNESS = FamilyConfig(
    group="fairness",
    schema=FairnessConfig,
    package_style="nested",
)

T = TypeVar("T", bound="BaseFairnessAssessor")


def register_fairness_adapter(
    *,
    algorithm_registry: Mapping[str, FairnessAdapterSemanticsHints],
    onnx_compatible_algorithms: frozenset[str] | _AllAlgorithmsSentinel = frozenset(),
    **common: Unpack[AdapterDecoratorOptions],
) -> Callable[[type[T]], type[T]]:
    def wrap(cls: type[T]) -> type[T]:
        cls.algorithm_registry = algorithm_registry  # type: ignore[misc]
        cls.ONNX_COMPATIBLE_ALGORITHMS = (  # type: ignore[misc]
            frozenset(algorithm_registry.keys())
            if onnx_compatible_algorithms is ALL
            else onnx_compatible_algorithms
        )
        return _register_core(cls, family=FAIRNESS, **common)

    return wrap
```

Pyright errors at decoration if `registry_name` (via `AdapterDecoratorOptions.Required`) or `algorithm_registry` is missing.

## 4. Schema (`configs/schema.py`)

Add a dataclass for the per-adapter config + a field on `AppConfig`:

```python
@dataclass
class FairnessConfig:
    _target_: str = MISSING
    # Overridden by the fairness config-group YAML (fairness=demographic_parity / ...).

# inside AppConfig:
    fairness: dict[str, FairnessConfig] = field(default_factory=dict)
```

The dict-of-configs shape mirrors transparency / robustness — multiple named adapters can coexist (`fairness.demographic_parity: ...`, `fairness.equalized_odds: ...`).

## 5. Factory (`fairness/factory.py`)

Iterate `config.fairness`, instantiate each entry via hydra-zen, run it. Mirror `src/raitap/robustness/factory.py` — same shape, swap module names.

## 6. Pipeline phase + result (`pipeline/phases/assess_fairness.py`, `fairness/report.py`, `pipeline/phases/registry.py`)

The pipeline is generic dispatch over `ASSESSMENT_PHASES` — you do **not** edit
`orchestrator.py` or `reporting/builder.py`. Three pieces:

**a. The phase work** (`pipeline/phases/assess_fairness.py`) — mirror `assess_transparency` / `assess_robustness`:

Each result **owns its visualisations** (`FairnessResult.visualisations`, a `list`
populated by the result's own `visualise()`); there is no parallel phase-level
list. So the work function returns just the results:

```python
def assess_fairness(config, model, data, forward_output, *, ...):
    """Run every adapter declared under ``config.fairness``."""
    results = []
    for name, adapter_config in config.fairness.items():
        result = ...  # instantiate via the factory, run, write artefacts
        result.visualise()  # populates result.visualisations
        results.append(result)
    return results
```

**b. The phase result** (`fairness/report.py`) — a `PhaseResult`: `Trackable` (how it logs) + `Reportable` (how it reports). Mirror `transparency/report.py` / `robustness/report.py`. It holds only the results and reaches each result's figures via `result.visualisations`:

```python
@dataclass
class FairnessPhaseResult(Trackable):
    fairness_results: list[FairnessResult] = field(default_factory=list)
    report_order: ClassVar[int] = 40  # orders sections vs metrics(10)/transparency(20)/robustness(30)

    def log(self, tracker, **kwargs) -> None:
        for result in self.fairness_results:
            result.log(tracker)
            for visualisation in result.visualisations:
                visualisation.log(tracker)

    def report_sections(self, ctx: ReportContext) -> tuple[ReportSection, ...]:
        ...  # iterate results -> result.visualisations, stage into ctx.assets_dir
```

**c. The phase class** lives in your module (`fairness/report.py`, alongside the result), subclassing `AssessmentPhase` from `raitap.pipeline.phases.base`:

```python
class FairnessPhase(AssessmentPhase):
    name = "fairness"

    def is_configured(self, config) -> bool:
        return bool(getattr(config, "fairness", None))

    def run(self, ctx) -> PhaseResult | None:
        return FairnessPhaseResult(fairness_results=list(assess_fairness(ctx.config, ...)))
```

**d. Register it** — the one pipeline edit (`pipeline/phases/registry.py`): import your phase + add it to the list:

```python
from raitap.fairness.report import FairnessPhase

ASSESSMENT_PHASES = (MetricsPhase(), TransparencyPhase(), RobustnessPhase(), FairnessPhase())
```

The configured-phase guard, tracker loop, and report builder all iterate
`ASSESSMENT_PHASES` / `RunOutputs.phase_results`, so they pick up the new phase
automatically. `report_order` decides where your sections land in the report.

## 7. Lazy module surface (`fairness/__init__.py`)

Mirror `src/raitap/transparency/__init__.py` — lazy `__getattr__` that resolves adapter names via `raitap._adapters.lookup("fairness", name)`. This is what powers `from raitap.fairness import demographic_parity`.

Also re-export `ALL` (`from raitap._adapters import ALL as ALL`) so adapter authors in this module can write `from raitap.fairness import ALL`.

## 8. Tests

- **Per adapter**: `src/raitap/fairness/assessors/tests/test_<adapter>.py` for each.
- **Registration smoke test**: `src/raitap/fairness/assessors/tests/test_registration.py` mirroring the existing one in `robustness/assessors/tests/`.
- **Partial-extras guard**: `src/raitap/fairness/tests/test_partial_extras_safe.py` mirroring the existing one in `robustness/tests/`.
- **E2E**: `src/raitap/fairness/tests/test_e2e_*.py` running the whole module end-to-end with real data.

## 9. Docs

- `docs/modules/fairness/index.md` — module landing page.
- `docs/modules/fairness/frameworks-and-libraries.md` — adapter + algorithm reference.
- Update `docs/index.md` or the top-level nav to link the new module.

## 10. pyproject.toml (optional)

If the module wraps third-party libraries, add a per-adapter extra and a `fairness` compound extra that pulls them all in:

```toml
[project.optional-dependencies]
demographic_parity = ["some-lib"]
equalized_odds = ["other-lib"]
fairness = ["raitap[demographic_parity,equalized_odds]"]
```

## Reference modules

When in doubt, copy from the most-similar existing module:

- **transparency** (`src/raitap/transparency/`) — pure-Python algorithms with rich per-call kwargs.
- **robustness** (`src/raitap/robustness/`) — same plus formal-verification subfamily (different abstract base).
- **metrics** (`src/raitap/metrics/`) — much smaller surface; useful if your module produces only summary numbers.

Same patterns, swap module names. The cross-cutting plumbing (`AdapterMixin`, `FamilyConfig`, `_register_core`) lives in `src/raitap/_adapters.py` and is module-agnostic — no edits needed there for a new module.
