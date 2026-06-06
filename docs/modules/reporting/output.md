---
title: "Output"
description: "RAITAP reports are compact summaries of the run, not a dump of every artifact written by the metrics, transparency, and robustness modules. The report builder first creates structured report content, then the default HTML renderer lays that"
myst:
  html_meta:
    "description": "RAITAP reports are compact summaries of the run, not a dump of every artifact written by the metrics, transparency, and robustness modules. The report builder first creates structured report content, then the default HTML renderer lays that"
---

# Output

RAITAP reports are compact summaries of the run, not a dump of every artifact
written by the metrics, transparency, and robustness modules. The report builder
first creates structured report content, then the default HTML renderer lays
that content out for browser viewing.

## Files

A reporting-enabled single run writes:

```text
reports/
├── report.html
├── report.zip
├── report_manifest.json
└── _assets/
    ├── ... native global or aggregated summary figures
    └── ... selected local sample figures
```

`report_manifest.json` records the semantic report structure, selected samples,
asset paths, and metadata used for sweep-level merging. The manifest is the
source of truth for merged reports. `report.html` is a standalone browser view
with embedded CSS. `report.zip` contains the HTML file, `report_manifest.json`,
and report-local `_assets` images for sharing. Use `reporting=pdf` for PDF output.

The original explainer artifacts are still kept under `transparency/` for
debugging and tracking. Report-local figures under `reports/_assets/` are the
curated subset used in the report.

## Report Structure

Generated reports use this structure:

1. **Executive Summary**
2. **Transparency Details**
3. **Robustness Details**
4. **Appendix**

Missing metrics, global explanations, aggregated explanations, and robustness sections are omitted. If no
local explanations are present, the transparency details render a short
placeholder rather than an empty card.

### Metrics

When metrics are enabled, this section includes the scalar metric table and any
metric figures produced by the metrics module.

### Global Explanations

This section contains true global content only:

- Native global visualisations backed by a dataset, population, or model-wide
  explanation contract.

RAITAP does not synthesize report-level global explanation artifacts from local
attribution tensors. This section is reserved for outputs produced directly by
the configured visualiser or underlying library with `GLOBAL` scope.

Current built-in transparency visualisers do not produce global outputs, so this
section is usually omitted.

### Aggregated Explanations

Aggregated explanations summarize the selected input batch. Current SHAP
bar, SHAP beeswarm, and tabular bar summaries belong here because they aggregate
local attribution values from the selected samples. RAITAP intentionally does
not call those figures global unless a future first-class dataset, population,
or model-wide contract proves that scope.

### Local Explanations

Local explanations are per-sample visualisations. RAITAP automatically selects a
small set of important examples so the report stays compact even when the run
contains a full batch or test set. By default, local details include up to three
selected samples.

The section is grouped by selected sample. Each sample starts with a sample
header group containing the input thumbnail and sample facts, followed by one
group per configured explainer visualiser. Those visualiser groups contain a
curated table with the explainer algorithm, semantic metadata, relevant
explainer parameters, and meaningful visualiser identity/rendering settings.
Display-only controls such as colorbar toggles and sample limits are omitted.

When the selected input modality can be rendered, the sample header thumbnail is
used as the shared original for that sample. Image explainers that normally
render an original-image panel next to their attribution are asked to render
attribution-only figures. If a thumbnail cannot be rendered for a selected
sample, that sample's visualiser figures keep their original panels.

Set
`reporting.show_original_per_explainer=true` for a more verbose local layout
where each explainer figure includes its own original input panel, samples use
overview/detail grouping, and no sample thumbnails are emitted.

Report-local asset names for compact local explanations use
`sample_<sample_index>_thumbnail_<n>.png` and
`sample_<sample_index>_<explainer>_<visualiser>.png`. The manifest schema is
unchanged, but tools that match asset filenames should account for this naming
pattern.

Compact empirical robustness report figures use
`robustness_<index>_<assessor>_sample_<sample_index>_<visualiser>.png`. The
robustness section stays grouped by assessor, but each group contains one figure
per selected report sample and visualiser. Duplicate clean-input and
perturbation-map panels are suppressed across configured visualisers. This
compact rendering is report-only: the `robustness/<assessor>/` artifacts and
their `metadata.json` visualiser references still point at the canonical
standalone PNGs. Set `reporting.show_redundant_robustness_panels=true` to reuse
the standalone robustness figures in the report.

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

Set `reporting.sample_selection` to pin the exact local samples shown in the
report. The option accepts sample IDs, filenames, or zero-based batch indices;
when it is set, RAITAP renders the requested samples in the configured order and
marks them as user-selected in `report_manifest.json`. Invalid, ambiguous,
duplicate, or out-of-range entries fail the run before expensive pipeline work
whenever the data metadata is available.

`reporting.sample_selection` is report-only. It does not reduce the data passed
through the model forward, metrics, explainers, visualisers, or tracking. To run
explainers on fewer samples, reduce `data.source` or use a future data-subsetting
feature.

## Hydra Multiruns

Each Hydra child run still writes its own `reports/report.html` and
`reports/report_manifest.json`. At the end of a multirun, RAITAP also creates
one merged report under the sweep directory:

```text
multirun/.../
├── 0/
│   └── reports/
│       ├── report.html
│       ├── report.zip
│       └── report_manifest.json
├── 1/
│   └── reports/
│       ├── report.html
│       ├── report.zip
│       └── report_manifest.json
└── reports/
    ├── report.html
    ├── report.zip
    ├── report_manifest.json
    └── _assets/
```

The merged report concatenates child manifest content in Hydra job order and
prefixes groups with the child job label and override summary. Missing child
manifests are skipped and recorded in the parent manifest metadata.

Set `reporting.multirun_report=false` to disable only the merged parent report while
keeping per-run reports.
