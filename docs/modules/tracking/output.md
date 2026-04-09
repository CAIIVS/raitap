# Output

Tracking does not create a dedicated local `tracking/` directory. Instead, the
tracking backend receives metadata, metrics, and copied artifacts from the run.

```text
tracking backend (for example MLflow)
├── params                          # Summary parameters such as assessment, model, and data fields
├── metrics                         # Scalar metrics such as performance.accuracy
└── artifacts/
    ├── config/config.json          # Full resolved RAITAP config
    ├── dataset.json                # Dataset description
    ├── metrics/...                 # Metrics artifacts forwarded from the local run directory
    ├── transparency/...            # Transparency artifacts forwarded from the local run directory
    └── model/...                   # Optional model artifact when tracking.log_model=true
```

For the current `MLFlowTracker`, scalar metrics are logged with the
`performance.` prefix, and transparency artifacts are stored under
`transparency/`.

The local Hydra run directory still exists. Tracking adds a backend-specific
record of that run rather than replacing the local files.
