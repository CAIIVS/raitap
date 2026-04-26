# Output

RAITAP reports are compact summaries of the run, not a dump of every artifact
written by the metrics and transparency modules. The report builder first creates
structured report content, then the PDF renderer lays that content out.

## Files

A reporting-enabled single run writes:

```text
reports/
├── report.pdf
├── report_manifest.json
└── _assets/
    ├── ... native global figures
    └── ... selected local sample figures
```

`report_manifest.json` records the semantic report structure, selected samples,
asset paths, and metadata used for sweep-level merging. The manifest is the
source of truth for merged reports; RAITAP does not stitch child PDFs together.

The original explainer artifacts are still kept under `transparency/` for
debugging and tracking. Report-local figures under `reports/_assets/` are the
curated subset used in the PDF.

## PDF Structure

Generated PDF reports use this section order:

1. **Metrics**
2. **Global Explanations**
3. **Local Explanations**

Empty sections are omitted. For example, a run without metrics will start with
global explanations if true global content exists, otherwise local explanations.

### Metrics

When metrics are enabled, this section includes the scalar metric table and any
metric figures produced by the metrics module.

### Global Explanations

This section contains true global content only:

- Native global visualisations, such as SHAP summary, bar, or beeswarm plots.

RAITAP does not synthesize report-level global explanation artifacts from local
attribution tensors. This section is reserved for outputs produced directly by
the configured visualiser or underlying library.

### Local Explanations

Local explanations are per-sample visualisations. RAITAP automatically selects a
small set of important examples so the report stays compact even when the run
contains a full batch or test set. By default, local details include up to three
selected samples.

The section contains:

- **Overview**: one shared most-relevant sample, rendered once per active local
  explainer so the visual comparison is meaningful.
- **Details**: selected important samples, grouped sample by sample, with one
  local visual from each active explainer.

For labeled classification outputs, RAITAP selects local detail samples in this
priority order:

1. Highest-confidence incorrect prediction.
2. Lowest-confidence prediction.
3. Highest-confidence correct prediction.

Duplicate samples are removed in priority order. For unlabeled classification
runs, selection uses confidence only. For non-classification or unsupported
output shapes, RAITAP falls back to the first available sample. Classification
confidence is the maximum softmax probability of the model output for each
sample.

## Hydra Multiruns

Each Hydra child run still writes its own `reports/report.pdf` and
`reports/report_manifest.json`. At the end of a multirun, RAITAP also creates one
merged report under the sweep directory:

```text
multirun/.../
├── 0/
│   └── reports/
│       ├── report.pdf
│       └── report_manifest.json
├── 1/
│   └── reports/
│       ├── report.pdf
│       └── report_manifest.json
└── reports/
    ├── report.pdf
    ├── report_manifest.json
    └── _assets/
```

The merged report concatenates child manifest content in Hydra job order and
prefixes groups with the child job label and override summary. Missing child
manifests are skipped and recorded in the parent manifest metadata.

Set `reporting.multirun_report=false` to disable only the merged parent report while
keeping per-run reports.
