---
title: "Architecture"
description: "The page explains the high level architecture of the raitap package. RAITAP is built to be modular."
myst:
  html_meta:
    "description": "The page explains the high level architecture of the raitap package. RAITAP is built to be modular."
---

# Architecture

The page explains the high level architecture of the `raitap` package. RAITAP is built to be modular.

The user launches RAITAP via the CLI. The `cli` module handles parsing, and passes over to the `deps` module, which handles parsing required dependencies and auto-adding them. Then the `pipeline` module orchestrates the assessment flow, delegating to the other modules, and finally terminating the run.

## Data flow

```{mermaid}
:alt: Data flow from CLI entry through deps bootstrap, Hydra composition, and the pipeline phases to reporting and tracking.
:caption: RAITAP data-flow — CLI to reporting.

flowchart TB
  A[uv run raitap …] --> B[raitap.cli.main]
  B -->|tracking stop?| Z[run_stop_command]
  B -->|--demo?| C1[load bundled demo.yaml]
  B -->|else| C2[parse user args]
  C1 --> D[raitap.deps.bootstrap.maybe_bootstrap]
  C2 --> D
  D -->|missing extras?| D1[uv sync + re-exec]
  D1 -->|re-exec| D
  D --> E[raitap.pipeline.__main__]
  E --> F["@hydra.main composes config"]
  F --> G[raitap.pipeline.orchestrator._run_pipeline]
  G --> H[forward · metrics · transparency · robustness]
  H --> I[reporting + tracking]
```

In prose: the CLI entry (`raitap.cli.main`) routes to either the tracking-stop subcommand or the standard flow. The standard flow loads the demo config (`--demo`) or parses user args, then calls `raitap.deps.bootstrap.maybe_bootstrap` to install missing extras and re-exec if needed. Once extras are pinned, control passes to `raitap.pipeline.__main__` (a `@hydra.main` entry that composes the config) then to `raitap.pipeline.orchestrator._run_pipeline`, which executes the forward / metrics / transparency / robustness phases in order before handing off to reporting and tracking. Programmatic callers skip the CLI + bootstrap and call `raitap.run(cfg, ...)` (defined in `raitap.api`) directly.

## Adapter registration

Every concrete adapter (explainer, assessor, metric, reporter, tracker, visualiser) self-registers via a namespaced facade decorator (`@adapters.transparency`, `@adapters.robustness`, `@adapters.metrics`, `@adapters.reporter`, `@adapters.tracker`, `@visualisers.transparency`, `@visualisers.robustness`). The decorator lives in `<module>/registration.py`; it delegates to `raitap._adapters._register_core`, which builds the hydra-zen builder, registers it with the `ConfigStore`, and populates `_BUILDERS` / `ADAPTER_EXTRAS` / `THIRD_PARTY_LIBS`. Lazy `__getattr__` on each family package resolves `raitap.<family>.<name>` via `raitap._adapters.lookup`.

See {doc}`adding-an-adapter` / {doc}`adding-an-algorithm` / {doc}`adding-a-module` for the contributor workflow.

## Visualisers and renderers

A **visualiser** owns the figure layout and is selected via YAML config (`@visualisers.transparency`). A **renderer** paints one attribution map onto a single axes and is auto-resolved from the explainer's library name at render time. If no renderer is registered for a library, a built-in dependency-free default is used. Plugin authors can register a custom renderer via `@adapters.image_renderer(for_library="<registry_name>")`.

## Directory structure

```text
src/
├── hydra_plugins/                  # Hydra-discovery namespace package
│   └── raitap_search_path.py       # re-exports RaitapSearchPathPlugin so Hydra finds it at import time
└── raitap/
    ├── __init__.py                 # lazy `__getattr__` resolves `raitap.<family>.<name>` via `_adapters.lookup`
    ├── __about__.py                # version + metadata constants
    ├── api.py                      # programmatic entry: `run(cfg, *, auto_install_deps=False, ...)`
    ├── cli.py                      # console-script entry; tracking-stop subcommand, --demo, help frame, then deps bootstrap
    ├── docs_preview.py             # `docs-preview` console-script: serves built Sphinx output for local preview
    ├── types.py                    # `Hardware` enum + small shared types
    ├── _adapters.py                # registration core: `AdapterMixin` (instance helpers), `FamilyConfig`,
    │                               # `AdapterDecoratorOptions` TypedDict, `_register_core`, `_BUILDERS`,
    │                               # `ADAPTER_EXTRAS`, `THIRD_PARTY_LIBS`, `ALL` sentinel, `discover`, `lookup`
    │
    ├── configs/                    # Hydra config tree shipped with the wheel
    │   ├── schema.py               # AppConfig + nested dataclasses; MISSING-typed required fields
    │   ├── searchpath.py           # RaitapSearchPathPlugin impl (appends pkg://raitap.configs)
    │   ├── utils.py                # ConfigStore registration of `raitap_schema`; resolve_run_dir helpers
    │   ├── zen.py                  # `register_zen_groups()`: hydra-zen ConfigStore writes for shapes
    │   │                           # the decorator can't express (`_target_: null`, `# @package _global_`)
    │   ├── adapter_factory.py      # short-name → full-target resolution (HTMLReporter, CaptumExplainer, …)
    │   ├── demo.yaml               # self-contained demo invoked by `raitap --demo`
    │   ├── reporting/{html,pdf,disabled}.yaml
    │   ├── transparency/{captum,shap}.yaml         # `_target_`-only stubs, @package-nested per library
    │   ├── robustness/{torchattacks,foolbox,marabou}.yaml
    │   ├── metrics/classification.yaml
    │   └── tracking/mlflow.yaml
    │
    ├── deps/                       # pre-pipeline dep inference + auto-sync (torch-free)
    │   │                           # The "torch-free" promise is what unlocks bootstrap-from-zero
    │   │                           # (bare `pip install raitap` → `raitap --demo`). It rests on
    │   │                           # every adapter family `__init__` staying free of top-level
    │   │                           # backend-lib imports — see `raitap.utils.lazy` for the contract
    │   │                           # and `deps/tests/test_bootstrap_from_zero.py` for the guard.
    │   ├── bootstrap.py            # maybe_bootstrap(): top-level flow + case A/B/C/D dispatch
    │   ├── inference.py            # walks composed config, picks extras from `_target_` mapping
    │   ├── availability.py         # reads raitap pyproject for declared extras + platform markers
    │   ├── conflicts.py            # enforces tool.uv.conflicts groups (torch-cpu vs torch-cuda, …)
    │   ├── probe.py                # host probe → cpu / cuda / xpu
    │   ├── python_version.py       # picks compatible Python interpreter for the selected extras
    │   ├── command.py              # renders the final uv-sync / uv-add / pip-install argv
    │   └── frame.py                # rich panels for deps status + error frames
    │
    ├── pipeline/                   # the assessment run, split by phase
    │   ├── __main__.py             # @hydra.main entry; composes config (incl. raitap_schema) → orchestrator._run_pipeline()
    │   ├── orchestrator.py         # _run_pipeline() + run_without_tracking() — runs configured phases; generic tracker loop over phase_results
    │   ├── ui.py                   # print_summary() — the rich panel banner
    │   ├── outputs.py              # RunOutputs (keyed phase_results + .metrics/.transparency/.robustness accessors) + PhaseResult + AdapterResult protocols
    │   └── phases/                 # cross-cutting phase infra only; module-specific work lives in each module's phase.py
    │       ├── base.py             # AssessmentPhase ABC + PhaseContext + run_adapters() (shared per-adapter loop)
    │       ├── registry.py         # ASSESSMENT_PHASES: single source of truth for the configured-phase guard + run loop
    │       ├── forward_pass.py     # forward_pass(): batched backend forward; extract_primary_tensor for dict/tuple outputs
    │       ├── prediction_summaries.py # prediction_summaries(): per-sample PredictionSummary rows from logits
    │       └── input_metadata.py   # input_metadata_for_data(): bridges raitap.data → transparency/robustness InputSpec
    │
    ├── models/                     # model loading + backend wrappers (PyTorch + ONNX)
    │   ├── model.py                # Model wrapper, source resolution (built-in name / .pt / state-dict / .onnx)
    │   ├── backend.py              # TorchBackend / OnnxBackend + hardware_label
    │   └── runtime.py              # resolve_torch_device, resolve_onnx_providers
    │
    ├── data/                       # dataset loading (images + tabular) + labels resolution
    │   ├── data.py                 # Data class; samples / tabular loaders
    │   └── samples.py              # bundled sample sets (imagenet_samples, mnist_samples, …) + labels CSVs
    │
    ├── metrics/                    # metrics adapters (currently torchmetrics-backed)
    │   ├── phase.py                # MetricsPhase + evaluate_metrics(): singleton phase (no adapter loop), infers num_classes
    │   ├── factory.py              # metrics_run_enabled, evaluate(), instantiation via adapter_factory
    │   ├── registration.py         # `@adapters.metrics` family decorator
    │   ├── base_metric_computer.py # `BaseMetricComputer` ABC + `MetricResult` dataclass
    │   ├── classification_metrics.py
    │   └── inputs.py               # target/prediction alignment, fallbacks when labels missing
    │
    ├── transparency/               # XAI adapters
    │   ├── phase.py                # TransparencyPhase + assess_transparency() (run_adapters) + resolve_explainer_runtime_kwargs + detection routing
    │   ├── explain_detection.py    # detection-task per-box K-loop (one ExplanationResult per detected box)
    │   ├── report.py               # TransparencyPhaseResult + Global/Aggregated/Local section builders
    │   ├── factory.py              # create_explainer / create_visualisers; runtime kwargs resolution
    │   ├── contracts.py            # ExplanationPayloadKind, InputSpec, MethodFamily, …
    │   ├── results.py              # ExplanationResult (AdapterResult: name/adapter_target/algorithm/semantics; owns its .visualisations) + Explanation object
    │   ├── explainers/             # CaptumExplainer, ShapExplainer + `registration.py`
    │   │                           # (`@adapters.transparency`) + `base_explainer.py`
    │   └── visualisers/            # CaptumImageVisualiser, ShapImageVisualiser, … + `registration.py`
    │                               # (`@visualisers.transparency`) + `base_visualiser.py`
    │
    ├── robustness/                 # adversarial-attack + formal-verification adapters
    │   ├── phase.py                # RobustnessPhase + assess_robustness() (run_adapters) + resolve_robustness_targets
    │   ├── report.py               # RobustnessPhaseResult + "Robustness" section builders
    │   ├── factory.py              # create_assessor; raitap-key migration warnings
    │   ├── results.py              # RobustnessResult (AdapterResult: name/adapter_target/algorithm/semantics; owns its .visualisations)
    │   ├── assessors/              # TorchattacksAssessor, FoolboxAssessor, MarabouAssessor +
    │   │                           # `registration.py` (`@adapters.robustness`) +
    │   │                           # `base_assessor.py`
    │   └── visualisers/            # ImagePairVisualiser, PerturbationHeatmapVisualiser, formal/* +
    │                               # `registration.py` (`@visualisers.robustness`)
    │
    ├── reporting/                  # HTML + PDF report builders + multirun aggregation
    │   ├── registration.py         # `@adapters.reporter` family decorator
    │   ├── base_reporter.py        # `BaseReporter` ABC
    │   ├── builder.py              # build_report(): generic dispatch — asks each phase result (Reportable) for its sections, ordered by report_order
    │   ├── staging.py              # shared figure/asset staging helpers (used by each phase's report renderer)
    │   ├── samples.py              # sample-selection model + strategies (phase-agnostic; keys off predictions)
    │   ├── html_reporter.py        # Jinja2-backed HTML renderer
    │   ├── pdf_reporter.py         # borb-backed PDF renderer
    │   ├── hydra_callback.py       # ReportingSweepCallback wired by reporting/{html,pdf}.yaml
    │   ├── sample_selection.py     # user-selected vs auto-picked sample logic
    │   ├── filenames.py            # output filename validation
    │   └── templates/              # Jinja templates
    │
    ├── tracking/                   # experiment-tracking adapters
    │   ├── registration.py         # `@adapters.tracker` family decorator
    │   ├── base_tracker.py         # BaseTracker abstract class + stop_detached hook
    │   ├── mlflow_tracker.py       # MLflow adapter
    │   ├── process_registry.py     # ~/.raitap/tracking_processes.json (used by `tracking stop`)
    │   └── stop.py                 # run_stop_command(): terminates detached tracker processes
    │
    ├── utils/                      # cross-cutting helpers
    │   ├── console.py              # rich panels (summary, failure, complete) + setup_logging
    │   ├── diagnostics.py          # Diagnostic dataclass + is_dev_install heuristic
    │   ├── errors.py               # RaitapError hierarchy
    │   ├── log.py                  # raitap_log wrapper (filename-aware warnings)
    │   ├── process.py              # cross-platform subprocess helpers
    │   └── tests/
    │
    └── tests/                      # cross-package integration tests (pipeline orchestration, memory leaks)
```
