# Understanding outputs

RAITAP writes its outputs to the Hydra run directory. By default, this is a
timestamped folder under `./outputs` relative to the directory where you launched RAITAP.

If needed, you can override the output location via
[global configuration](configuration/global-config-options.md).

## Default layout

The default layout looks like this:

```text
outputs/
└── 2026-02-28/
    └── 14-30-45/
        ├── attributions.pt
        ├── <VisualiserName>.png
        └── metadata.json
```

## Main files

- `attributions.pt`: the raw attribution tensor produced by the transparency module.
- `<VisualiserName>.png`: one image file per generated visualisation.
- `metadata.json`: a snapshot of the resolved run configuration and metadata.

The exact contents depend on the modules you enabled. For example, runs with tracking or
metrics may produce additional outputs.
