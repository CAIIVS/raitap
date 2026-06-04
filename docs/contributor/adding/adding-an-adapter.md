---
title: "Adding an adapter"
description: "Adapters (explainers, assessors, metrics, reporters, trackers, visualisers) self-register via a namespaced facade decorator (@adapters.transparency, @adapters.robustness, ...). Adding one is a single file plus optional pyproject.toml + test + docs entries. No central registry to edit."
myst:
  html_meta:
    "description": "Adapters (explainers, assessors, metrics, reporters, trackers, visualisers) self-register via a namespaced facade decorator (@adapters.transparency, @adapters.robustness, ...). Adding one is a single file plus optional pyproject.toml + test + docs entries. No central registry to edit."
---

# Adding an adapter

Adapters let RAITAP delegate to other libraries. Each family (transparency, robustness, metrics, reporting, tracking, visualisers) exposes its own registration decorator; you drop that decorator on a class, implement the abstract methods, and you are done. There is no central registry file to edit. `registry_name` is the only required kwarg and pyright errors at the decoration site if you forget it.

The walkthrough below uses a fictional `superxai-lib` transparency explainer.

## 1. Find the relevant module where to create the new adapter

RAITAP is organised into modules under `src/raitap/`, each owning the adapters for one family. Ideally a library fits into a single module. If it genuinely spans two, ship two adapters (one per module) and use `algorithm_registry` to keep their responsibilities disjoint.

## 2. Create the new adapter file

Name the file after the library (e.g. `superxai_explainer.py`). Then:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from raitap import adapters
from raitap.transparency.contracts import ExplainerSemanticsHints, MethodFamily

from .base_explainer import AttributionOnlyExplainer

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


@adapters.transparency(
    registry_name="superxai",      # CLI `+transparency=superxai` / Python `from raitap.transparency import superxai`
    # extra="superxai",            # uv extra name; defaults to `registry_name` (omit unless they differ, see metrics for an exception)
    library="superxai-lib",        # real PyPI package name; drives `self._lazy_import()`
    error_patterns={               # rewrite cryptic upstream errors at call sites
        r"some library footgun": "Do X instead.",
    },
    suppress_warnings=[            # optional library-noise filters installed at decoration time
        (r"some noisy.*pattern", UserWarning, r"superxai.*"),
    ],
    algorithm_registry={           # required kwarg, pyright errors at decoration if missing
        # supertreeshap is model-agnostic (works on any backend): leave requires default empty.
        "supertreeshap": ExplainerSemanticsHints({MethodFamily.SHAPLEY}),
    },
    # output_payload_kind=ExplanationPayloadKind.ATTRIBUTIONS  # optional, this is the default
)
class SuperXAIExplainer(AttributionOnlyExplainer):
    def __init__(self, algorithm: str, **init_kwargs):
        super().__init__()
        self.algorithm = algorithm
        self.init_kwargs = init_kwargs

    # check_backend_compat: inherited from AdapterMixin. Raises BackendIncompatibilityError
    # when algorithm.requires - backend.provides is non-empty. Write zero gate code here.
    # Override only for a non-capability contract, e.g. MarabouAssessor per-call setup.

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

- **Base class.** `AttributionOnlyExplainer` provides batching, artefact persistence, and `explain()` orchestration. Other families use other bases (`EmpiricalAttackAssessor` for robustness, `BaseMetricComputer` for metrics, etc.); find them in files starting with `base_`. The concrete adapter is just a normal class; nothing flags it as abstract (the old `abstract=True` workaround was removed).
- **Decorator.** `@adapters.transparency(...)` is the sole entry point for registration. `registry_name` is required and pyright-checked at the decoration site via `Required[str]`. Each family has its own facade attribute (`adapters.robustness`, `adapters.metrics`, `adapters.reporter`, `adapters.tracker`, `visualisers.transparency`, `visualisers.robustness`): pick the one matching your base class.
- **Registration kwargs.** `library` is the pip name powering `self._lazy_import()`: pass it when you wrap a third-party package (the usual case). `extra` is the uv extra surfaced in install hints and scanned by `raitap.deps.inference`; it **defaults to `registry_name`** so you only need to set it explicitly when they differ (e.g. `classification_metrics` + `detection_metrics` both share `extra="metrics"`). `error_patterns` and `suppress_warnings` are optional polish.
- **`algorithm_registry` (decorator kwarg).** Transparency and robustness only. Maps algorithm name to a per-algorithm semantics-hints value RAITAP tracks and reports on (`ExplainerSemanticsHints` for transparency, `AssessorSemanticsHints` for robustness). **Required**: pyright errors at the decoration site if you omit it. Missing or misnamed entries make algorithms unselectable. The decorator assigns it onto the class so `type(self).algorithm_registry` still works at runtime.
- **`output_payload_kind` (decorator kwarg).** Transparency only. Tells the report renderer what artefact shape the explainer emits (`ATTRIBUTIONS`, `SALIENCY_MAP`, ...). Defaults to `ExplanationPayloadKind.ATTRIBUTIONS`; only pass it if your explainer emits something else.
- **Backend compatibility.** Inherited `check_backend_compat` (from `AdapterMixin`) raises `BackendIncompatibilityError` when `algorithm.requires - backend.provides` is non-empty. You write zero gate code. Gradient-based algorithms declare `requires={Capability.AUTOGRAD}` on their `ExplainerSemanticsHints` / `AssessorSemanticsHints` entry; model-agnostic algorithms (SHAP KernelExplainer, Occlusion, FeatureAblation) leave `requires` at its default empty frozenset and run on any backend including ONNX. Override `check_backend_compat` only for a non-capability contract (Marabou uses it for per-call setup; auto-LiRPA calls `super()` then warns on XPU). Import: `from raitap.utils.errors import BackendIncompatibilityError` (also re-exported from `raitap.robustness` and `raitap.transparency`). See {doc}`../capabilities` for what each capability means.
- **`super().__init__()`.** Cooperative parent init: the base class allocates buffers the framework reads later (e.g. `self.attributions = None`). Forgetting raises `AttributeError` deep inside `explain()`. Always call it first when overriding `__init__`.
- **`self._lazy_import()`.** Inherited from `AdapterMixin`. Imports `library` (or `f"{library}.{submodule}"` if you pass `submodule=`) at call time, keeping `import raitap` cheap and letting users install RAITAP without every wrapped library. Raises a clear install-hint `ImportError` if the library is missing.
- **Backend libs (`torch`, `torchvision`, `onnxruntime`) need `lazy_import` too.** If your adapter file uses `torch.Tensor` / `torch.nn` / etc, do NOT add `import torch` at module top-level. Use the `from raitap.utils.lazy import lazy_import` pattern (see that module's docstring for the `TYPE_CHECKING` + `lazy_import("torch")` recipe). This preserves the bootstrap-from-zero promise: `raitap.deps.bootstrap._compose` walks every adapter `__init__` on a bare venv (no torch installed yet) to infer extras before it installs them. A top-level `import torch` breaks the whole bootstrap. The per-family `tests/test_partial_extras_safe.py` poisons `torch` in `sys.modules` to catch regressions immediately.
- **`self._rethrow()`.** Inherited context manager. Catches exceptions from the wrapped library and rewrites known-cryptic ones using your `error_patterns` map.
- **Abstract methods.** `AttributionOnlyExplainer` → `compute_attributions(...)`. `EmpiricalAttackAssessor` → `generate_adversarial(...)`. `BaseMetricComputer` → `compute() -> MetricResult`. Check the base file for exact signatures.

## 3. Update the pyproject.toml file

Add the per-library extra and chain it into the module's compound extra. Use the extra name in the chain, not the PyPI name, if they differ.

```toml
[project.optional-dependencies]

# Transparency module
# ...
superxai = ["superxai-lib"]

transparency = ["raitap[shap,captum,superxai]"]
```

## 4. Update the tests

Tests go in `tests/raitap/<module>/<subdir>/test_<name>_<entity>.py`. Also add an E2E test, and update the E2E test matrix if the module has one.

Cover at minimum: registry membership, `check_backend_compat`, decorator wiring (the class lands in `_BUILDERS["<group>"]["<name>"]` and in `ADAPTER_EXTRAS`), and one `compute_attributions` / `generate_adversarial` / `compute` happy path. CI enforces module coverage.

No stub workaround needed. Concrete adapters are just concrete classes, and the decorator carries every family-required piece of metadata (`algorithm_registry`, `output_payload_kind`). A test stub is just `@adapters.<family>(registry_name="_stub", algorithm_registry={...}, ...)` over a minimal subclass implementing the abstract methods.

## 5. Update the docs

Add a row to `docs/modules/<module>/frameworks-and-libraries.md` documenting the algorithms you support (or the analogous page for non-adapter modules). YAML + Python tabs both required via `config-tabs`. This is what users see when they look up "does raitap support X?".

## Adapter families and where to look

| Module                 | Abstract base                                                                                                                               | Registration decorator                                              | Group              | Schema               |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------ | -------------------- |
| Transparency explainer | `raitap.transparency.explainers.base_explainer.AttributionOnlyExplainer` / `FullExplainer`                                                  | `@adapters.transparency`                                            | `transparency`     | `TransparencyConfig` |
| Robustness assessor    | `raitap.robustness.assessors.base_assessor.EmpiricalAttackAssessor` / `FormalVerificationAssessor`                                          | `@adapters.robustness`                                              | `robustness`       | `RobustnessConfig`   |
| Metric                 | `raitap.metrics.base_metric_computer.BaseMetricComputer`                                                                                    | `@adapters.metrics`                                                 | `metrics`          | `MetricsConfig`      |
| Reporter               | `raitap.reporting.base_reporter.BaseReporter`                                                                                               | `@adapters.reporter`                                                | `reporting`        | `ReportingConfig`    |
| Tracker                | `raitap.tracking.base_tracker.BaseTracker`                                                                                                  | `@adapters.tracker`                                                 | `tracking`         | `TrackingConfig`     |
| Visualiser             | `raitap.transparency.visualisers.base_visualiser.BaseVisualiser` / `raitap.robustness.visualisers.base_visualiser.BaseRobustnessVisualiser` | `@visualisers.transparency` / `@visualisers.robustness`             | (no Hydra group) |                    |

## Adding a new algorithm to an existing adapter

If the library already has an adapter and you just want to expose a new algorithm, add an entry to that adapter's `algorithm_registry` decorator kwarg. Set `requires={Capability.AUTOGRAD}` on the hint if the algorithm needs gradients; leave `requires` at default empty if it is model-agnostic. No new file, no pyproject.toml change. Add a unit test that constructs the adapter with the new algorithm and asserts a successful happy-path call.

## When the registration decorators aren't enough

A handful of Hydra shapes can't be expressed via the registration decorators alone and live as direct `ConfigStore` writes in `src/raitap/configs/zen.py::register_zen_groups`:

- **`# @package _global_` injections.** E.g. the `reporting=html` / `reporting=pdf` entries push both the `reporting` node **and** a `hydra.callbacks.reporting_sweep` block into the root config so multirun report aggregation auto-wires. The decorators always write under `package="<group>"` or `"<group>.<name>"`; they can't reach `_global_`.
- **`_target_: null` variants.** E.g. `reporting=disabled` carries `_target_: null` + `multirun_report: false`. The decorators always target a concrete class.
- **Custom hydra-zen `to_config=` or non-dataclass nodes.** Same escape hatch.

If your new adapter only needs to set `_target_` + optional kwargs on its schema (the 95% case), don't touch `zen.py`. Use the decorator, done. If you genuinely need one of the special shapes above, add a `cs.store(group=..., name=..., package=..., node=...)` block in `register_zen_groups` after the `store.add_to_hydra_store(...)` flush: that order lets your specialisation overwrite the decorator-generated entry.
