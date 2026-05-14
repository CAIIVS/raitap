# Architecture

The page explains the high level architecture of the `raitap` package. RAITAP is built to be modular.

The user launches RAITAP via the CLI. The `cli` module handles parsing, and passes over to the `deps` module, which handles parsing required dependencies and auto-adding them. Then the `pipeline` module orchestrates the assessment flow, delegating to the other modules, and finally terminating the run.

## Data flow

```{mermaid}
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
  F --> G[raitap.pipeline.pipeline.run]
  G --> H[forward · metrics · transparency · robustness]
  H --> I[reporting + tracking]
```

## Directory structure

```text
src/
├── hydra_plugins/                  # Hydra-discovery namespace package
│   └── raitap_search_path.py       # re-exports RaitapSearchPathPlugin so Hydra finds it at import time
└── raitap/
    ├── cli.py                      # console-script entry; tracking-stop subcommand, --demo, help frame, then deps bootstrap
    ├── docs_preview.py             # `docs-preview` console-script: serves built Sphinx output for local preview
    ├── registry_base.py            # WithAlgorithmRegistry generic + base hooks shared by transparency/robustness
    │
    ├── configs/                    # Hydra config tree shipped with the wheel
    │   ├── schema.py               # AppConfig + nested dataclasses; MISSING-typed required fields
    │   ├── searchpath.py           # RaitapSearchPathPlugin impl (appends pkg://raitap.configs)
    │   ├── utils.py                # ConfigStore registration of `raitap_schema`; resolve_run_dir helpers
    │   ├── adapter_factory.py      # short-name → full-target resolution (HTMLReporter, CaptumExplainer, …)
    │   ├── demo.yaml               # self-contained demo invoked by `raitap --demo`
    │   ├── reporting/{html,pdf,disabled}.yaml
    │   ├── transparency/{captum,shap}.yaml         # `_target_`-only stubs, @package-nested per library
    │   ├── robustness/{torchattacks,foolbox,marabou}.yaml
    │   ├── metrics/classification.yaml
    │   └── tracking/mlflow.yaml
    │
    ├── deps/                       # pre-pipeline dep inference + auto-sync (torch-free)
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
    │   ├── __main__.py             # @hydra.main entry; composes config (incl. raitap_schema) → orchestrator.run()
    │   ├── orchestrator.py         # run() + run_without_tracking() — wires the phases together; tracker context
    │   ├── ui.py                   # print_summary() — the rich panel banner
    │   ├── outputs.py              # PredictionSummary + RunOutputs dataclasses (typed return)
    │   └── phases/                 # one file per phase; filename matches the public function
    │       ├── forward_pass.py     # forward_pass(): batched backend forward; extract_primary_tensor for dict/tuple outputs
    │       ├── evaluate_metrics.py # evaluate_metrics(): runs metrics when configured, infers num_classes
    │       ├── assess_transparency.py  # assess_transparency(): instantiates explainers; resolve_explainer_runtime_kwargs (auto_pred)
    │       ├── assess_robustness.py    # assess_robustness(): instantiates assessors; resolve_robustness_targets (labels or argmax fallback)
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
    │   ├── factory.py              # metrics_run_enabled, evaluate(), instantiation via adapter_factory
    │   ├── classification_metrics.py
    │   └── inputs.py               # target/prediction alignment, fallbacks when labels missing
    │
    ├── transparency/               # XAI adapters
    │   ├── factory.py              # create_explainer / create_visualisers; runtime kwargs resolution
    │   ├── contracts.py            # ExplanationPayloadKind, InputSpec, MethodFamily, …
    │   ├── results.py              # ExplanationResult + Explanation orchestration object
    │   ├── explainers/             # CaptumExplainer, ShapExplainer (subclass of base_explainer)
    │   └── visualisers/            # CaptumImageVisualiser, ShapImageVisualiser, InputThumbnailVisualiser, …
    │
    ├── robustness/                 # adversarial-attack + formal-verification adapters
    │   ├── factory.py              # create_assessor; raitap-key migration warnings
    │   ├── assessors/              # TorchattacksAssessor, FoolboxAssessor, MarabouAssessor
    │   └── visualisers/            # ImagePairVisualiser, PerturbationHeatmapVisualiser, formal/*
    │
    ├── reporting/                  # HTML + PDF report builders + multirun aggregation
    │   ├── builder.py              # build_report(): assembles sections from RunOutputs
    │   ├── html_reporter.py        # Jinja2-backed HTML renderer
    │   ├── pdf_reporter.py         # borb-backed PDF renderer
    │   ├── hydra_callback.py       # ReportingSweepCallback wired by reporting/{html,pdf}.yaml
    │   ├── sample_selection.py     # user-selected vs auto-picked sample logic
    │   ├── filenames.py            # output filename validation
    │   └── templates/              # Jinja templates
    │
    ├── tracking/                   # experiment-tracking adapters
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
