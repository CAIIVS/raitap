---
title: "Adding an adapter"
description: "Adapters (explainers, assessors, metrics, reporters, trackers) self-register via raitap.adapters.AdapterMixin. Adding a new one is one file plus optional pyproject.toml + test + docs entries. No central registry to edit."
myst:
  html_meta:
    "description": "Adapters (explainers, assessors, metrics, reporters, trackers) self-register via raitap.adapters.AdapterMixin. Adding a new one is one file plus optional pyproject.toml + test + docs entries. No central registry to edit."
---

# Adding an adapter

Adapters (explainers, assessors, metrics, reporters, trackers) allow RAITAP to delegate to other libraries. To add one, you simply need to create a new file, follow a few rules and then update some package files.

The following sections will guide you. We will imagine we adding a new explainer for the `superxai-lib` library.

## 1. Find the relevant module where to create the new adapter

RAITAP is organised into modules, each containing a set of related adapters. The modules are located in the `src/raitap/` directory. Ideally, a library fits clearly into a single module. However, some libraries might span multiple. In such cases, you should create 2 separate adapters in each module, and carefully prevent one adapter from calling functions that fit into the other (e.g. using `algorithm_registry`, see below).

## 2. Create the new adapter file

1. Name your file by following the pattern exposed by other adapter files.
2. Create a class as follows. We take the example of the transparency module:

    ```python
    class SuperXAIExplainer(
        AttributionOnlyExplainer,
        registry_name="superxai",         # CLI `+transparency=superxai` / Python `from raitap.transparency import superxai`
        extra="superxai",                 # uv extra as named in the pyproject.toml file, see below
        library="superxai-lib",           # real PyPI package name
        error_patterns={                  # mapping between cryptic original library errors and a more user-friendly RAITAP one
            re.compile(r"some library footgun"): "Do X instead.",
            re.compile(r"some other library footgun"): "Do Y instead.",
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

    - We inherit from the `AttributionOnlyExplainer` abstract base class. For other modules, the name might differ (e.g. `EmpiricalAttackAssessor` for the robustness module). You will usually find it in a file which name starts with `base_`.
    - We set the required class-keyword arguments, as defined by `AdapterMixin`: `registry_name`, `extra`, `library`. Some are optional but useful, such as `error_patterns` and `suppress_warnings`.
    - For modules that require the user specifying a specific algo (transparency, robustness,...), we declare the `algorithm_registry` class dictionary. It maps the algorithm name to the method families it supports., which is extremely important so RAITAp can track and report about the results.
    - We implement the methods required by the abstract base class. For example, `AttributionOnlyExplainer` requires the user to implement the `compute_attributions` method.

## 3. Update the pyproject.toml file

1. Add the `superxai` extra to the `pyproject.toml` file. This is used to install the `superxai-lib` PyPI package when the user composes a config that needs it. Ensure the name you choose matches the values you set in the adapter class.

    ```toml
    [project.optional-dependencies]
    # ...

    # Transparency module
    # ...
    superxai = ["superxai-lib"]

    # ...
    ```

2. Update the module's compound extra. Note that you should use the extra's name, not the PyPI package name, if they differ (like in our example).

     ```toml
    [project.optional-dependencies]
    
    # ...

    # Transparency module
    # ...

    superxai = ["superxai-lib"]

    # ...

    transparency = ["raitap[shap,captum,superxai]"]

    # ...
    ```

## 4. Update the tests

Write tests for the new adapter. The tests should be located in the `tests/raitap/<module>/<subdir>/test_<name>_<entity>.py` file. Also create E2E tests, and if relevant for the modified module, update the E2E test matrix.

Cover at minimum: registry membership, `check_backend_compat`, `__init_subclass__` wiring (the class lands in `_BUILDERS["<group>"]["<name>"]` and in `ADAPTER_EXTRAS`), and one `_compute` / `generate_adversarial` / `compute` happy path. CI enforces coverage on the module.

## 5. Update the docs

Add a row to the `docs/modules/<module>/frameworks-and-libraries.md` file documenting the algorithms you support (or analogous page for non-adapter modules). YAML + Python tabs both required via `config-tabs`. This is what users see when they look up "does raitap support X?".

## Adapter families and where to look

| Module                 | Abstract base                                                                                                                               | Group              | Schema               |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | -------------------- |
| Transparency explainer | `raitap.transparency.explainers.base_explainer.AttributionOnlyExplainer` / `FullExplainer`                                                  | `transparency`     | `TransparencyConfig` |
| Robustness assessor    | `raitap.robustness.assessors.base_assessor.EmpiricalAttackAssessor` / `FormalVerificationAssessor`                                          | `robustness`       | `RobustnessConfig`   |
| Metric                 | `raitap.metrics.base_metric.BaseMetricComputer`                                                                                             | `metrics`          | `MetricsConfig`      |
| Reporter               | `raitap.reporting.base_reporter.BaseReporter`                                                                                               | `reporting`        | `ReportingConfig`    |
| Tracker                | `raitap.tracking.base_tracker.BaseTracker`                                                                                                  | `tracking`         | `TrackingConfig`     |
| Visualiser             | `raitap.transparency.visualisers.base_visualiser.BaseVisualiser` / `raitap.robustness.visualisers.base_visualiser.BaseRobustnessVisualiser` | — (no Hydra group) | —                    |

## When `AdapterMixin` isn't enough

A handful of Hydra shapes can't be expressed via `AdapterMixin` alone and live as direct `ConfigStore` writes in `src/raitap/configs/zen.py::register_zen_groups`:

- **`# @package _global_` injections** — e.g. the `reporting=html` / `reporting=pdf` entries push both the `reporting` node **and** a `hydra.callbacks.reporting_sweep` block into the root config so multirun report aggregation auto-wires. The mixin always writes under `package="<group>"` or `"<group>.<name>"`; it can't reach `_global_`.
- **`_target_: null` variants** — e.g. `reporting=disabled` carries `_target_: null` + `multirun_report: false`. The mixin always targets a concrete class.
- **Anything that needs a custom hydra-zen `to_config=` or a non-dataclass node** — same escape hatch.

If your new adapter only needs to set `_target_` + optional kwargs on its schema (the 95% case), don't touch `zen.py`. Use class kwargs, done. If you genuinely need one of the special shapes above, add a `cs.store(group=..., name=..., package=..., node=...)` block in `register_zen_groups` after the `store.add_to_hydra_store(...)` flush — that order lets your specialisation overwrite the mixin-generated entry.
