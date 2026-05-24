# `raitap` package layout

The RAITAP package is organised as a Hydra-driven pipeline. The CLI / Python
API resolve a config, the orchestrator runs a sequence of **phases**, and each
phase drives one **adapter family** (transparency, robustness, metrics,
reporting, tracking). Adapters self-register via decorators in `_adapters.py`,
so adding one never requires editing a central registry.

```text
src/raitap/
├── __about__.py            # package version (from installed metadata)
├── __init__.py             # public re-exports
├── api.py                  # public Python API — raitap.run(...) facade over the orchestrator
├── cli.py                  # console-script entry: subcommand dispatch, --demo, deps bootstrap
├── _adapters.py            # single registration point; family decorators → hydra-zen builds() + lazy attrs
├── types.py                # root-level enum aliases (Hardware, Task) — import-light, no torch
├── docs_preview.py         # `raitap docs` — Sphinx live-reload preview server
├── py.typed                # PEP 561 marker (ships type hints)
│
├── configs/                # Hydra / hydra-zen config layer
│   ├── schema.py           # structured-config dataclasses (the RAITAP config schema)
│   ├── adapter_factory.py  # builds adapters from resolved config
│   ├── searchpath.py       # config search-path plugin
│   ├── zen.py              # hydra-zen helpers
│   ├── demo.yaml           # bundled --demo config
│   ├── hydra/              # hydra runtime config (launcher/, etc.)
│   └── extras/             # optional config fragments
│
├── deps/                   # dependency inference + auto-install (runs first, pre-torch)
│   ├── inference.py        # decide which extras a config needs
│   ├── bootstrap.py        # sync extras + re-exec
│   ├── availability.py     # runtime "is X installed" probes
│   ├── static_scan.py      # static config scan
│   └── command.py / conflicts.py / frame.py / probe.py / python_version.py
│
├── data/                   # dataset loading, preprocessing, sample selection
│   ├── data.py             # dataset entry
│   ├── preprocessing.py    # transforms
│   ├── samples.py          # sample selection
│   ├── metadata.py         # input metadata
│   └── types.py / utils.py
│
├── models/                 # load any model into a uniform ModelBackend  (see models/README.md)
│   ├── backend.py          # ModelBackend abstraction consumed downstream
│   ├── model.py            # loader (torchvision / ONNX / .pt|.pth flavours)
│   ├── runtime.py          # inference runtime
│   └── task_wrappers.py    # classification / detection task wrappers
│
├── pipeline/               # orchestration
│   ├── orchestrator.py     # runs the phase sequence
│   ├── outputs.py          # output artifacts
│   ├── ui.py               # progress / console UI
│   ├── __main__.py         # `python -m raitap.pipeline`
│   └── phases/             # one file per phase, run in order
│       ├── forward_pass.py
│       ├── input_metadata.py
│       ├── prediction_summaries.py
│       ├── evaluate_metrics.py
│       ├── assess_transparency.py
│       ├── assess_robustness.py
│       └── explain_detection.py
│
├── transparency/           # attribution / explainability family  (see transparency/README.md)
│   ├── factory.py          # build explainers from config
│   ├── contracts.py / semantics.py / results.py
│   ├── algorithm_allowlist.py
│   ├── explainers/         # captum, shap, custom, full
│   └── visualisers/        # heatmaps, thumbnails, detection-image, tabular
│
├── robustness/             # adversarial + formal robustness family
│   ├── factory.py
│   ├── contracts.py / semantics.py / results.py
│   ├── assessors/          # foolbox, torchattacks (empirical) + marabou (formal)
│   └── visualisers/
│       ├── empirical/      # image-pair, perturbation-heatmap
│       └── formal/         # output-bounds plots, verdict summary
│
├── metrics/                # task metrics (classification / detection)
│   ├── factory.py
│   ├── classification_metrics.py / detection_metrics.py
│   ├── base_metric_computer.py / inputs.py
│   └── visualizers.py
│
├── reporting/              # HTML / PDF report generation  (templated)
│   ├── builder.py          # assemble report
│   ├── view_model.py       # data → template model
│   ├── html_reporter.py / pdf_reporter.py
│   ├── sections.py / sample_selection.py / manifest.py / filenames.py
│   ├── hydra_callback.py   # fires reporting at run end
│   └── templates/          # Jinja templates
│
├── tracking/               # experiment tracking  (see tracking/README.md)
│   ├── mlflow_tracker.py
│   ├── process_registry.py / stop.py   # backs `raitap tracking stop`
│   └── base_tracker.py
│
├── testing/                # shared test helpers / fixtures
└── utils/                  # cross-cutting helpers
    ├── lazy.py             # lazy imports (keep heavy deps out of import path)
    ├── log.py / console.py / colour.py
    ├── errors.py / diagnostics.py
    └── serialization.py / process.py / status_frame.py
```

## Conventions

- **Adapter families**: Modules that call 3rd party libraries (`transparency`, `robustness`, `metrics`, `reporting`,
  `tracking`) use the adapter pattern: `factory.py` + `registration.py` +
  `base_*.py` + concrete adapters in a subpackage. Each adapter registers
  itself through the matching decorator in `_adapters.py`.
- **`tests/`** subdirs is colocated to the code they cover, in every module.
- **Import weight matters.** `types.py` and `utils/lazy.py` exist so that
  importing config/CLI code doesn't drag in torch before the deps bootstrap
  has had a chance to sync it. Do not add non-lazy imports or the entire pipeline will break.
