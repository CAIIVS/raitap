# Understanding outputs

RAITAP writes its outputs to the Hydra run directory. By default, this is a
timestamped folder under `./outputs` relative to the directory where you launched RAITAP.

If needed, you can override the output location via
{doc}`configuration/global-config-options`.

Here is the output of an illustrative example run:

```text
outputs/                                      # Hydra's default output directory
└── 2026-02-28/                               # Run date
    └── 14-30-45/                             # Run time
        ├── __main__.log                      # Run log
        ├── .hydra/                           # Hydra configuration logs
        │   ├── config.yaml
        │   ├── hydra.yaml
        │   └── overrides.yaml
        ├── metrics/                          # Metric module outputs (see below for details)
        │   ├── artifacts.json
        │   ├── metadata.json
        │   └── metrics.json
        ├── reports/                          # PDF report and report-only assets
        │   ├── report.pdf
        │   ├── report_manifest.json
        │   └── _assets/
        └── transparency/                     # Transparency module outputs (see below for details)
            ├── explainerA/                    
            │   ├── attributions.pt           
            │   ├── visualisation1.png
            │   ├── visualisation2.png
            │   └── metadata.json              
            └── explainerB/                    
                ├── attributions.pt           
                ├── visualisation1.png
                └── metadata.json              
```

The `transparency/` directory keeps the full explainer artifacts for debugging
and tracking. The `reports/` directory contains the compact report, its
`report_manifest.json`, and curated report-only figure assets.

You may want to look at each module's output directory for more details:

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} Transparency
:link: ../../modules/transparency/output
:link-type: doc
:::

:::{grid-item-card} Metrics
:link: ../../modules/metrics/output
:link-type: doc
:::

:::{grid-item-card} Reporting
:link: ../../modules/reporting/output
:link-type: doc
:::

::::
