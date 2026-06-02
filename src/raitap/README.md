The RAITAP package is organised as a Hydra-driven pipeline. The CLI / Python
API resolve a config, the orchestrator runs a sequence of **phases**, and each
phase drives one **adapter family** (transparency, robustness, metrics,
reporting, tracking). Adapters self-register via decorators in `_adapters.py`,
so adding one never requires editing a central registry.

```text
src/raitap/
├── api.py / cli.py      # entry points: public run() facade + console script
├── _adapters.py         # single registration point for all adapter families
├── types.py             # root enum aliases (import-light, no torch)
│
├── configs/             # Hydra / hydra-zen config layer (schema, demo, search path)
├── deps/                # dependency inference + auto-install (runs first, pre-torch)
├── data/                # dataset loading, preprocessing, sample selection
├── models/              # load any model into a uniform ModelBackend
├── pipeline/            # orchestrator + phases/ (shared phase infra: base, registry, run_adapters, forward pass)
├── transparency/        # attribution / explainability family
├── robustness/          # adversarial + formal robustness family
├── metrics/             # task metrics (classification / detection)
├── reporting/           # HTML / PDF report generation (Jinja templates)
├── tracking/            # experiment tracking (mlflow)
├── testing/             # shared test helpers / fixtures
└── utils/               # cross-cutting helpers (lazy imports, logging, errors)
```

## Conventions

- **Adapter families**: Modules that call 3rd party libraries (`transparency`, `robustness`, `metrics`, `reporting`,
  `tracking`) use the adapter pattern: `factory.py` + `registration.py` +
  `base_*.py` + concrete adapters in a subpackage. Each adapter registers
  itself through the matching decorator in `_adapters.py`.
- **Phase entry point**: each assessment family (`metrics`, `transparency`,
  `robustness`) owns a `phase.py` — its `AssessmentPhase` subclass + work
  function — which the orchestrator assembles via `pipeline/phases/registry.py`.
  Start there to trace a run. `pipeline/phases/` itself holds only cross-cutting
  infra, so adding a module touches it for exactly one registry line.
- **`tests/`** subdirs is colocated to the code they cover, in every module.
- **Import weight matters.** `types.py` and `utils/lazy.py` exist so that
  importing config/CLI code doesn't drag in torch before the deps bootstrap
  has had a chance to sync it. Do not add non-lazy imports or the entire pipeline will break.
