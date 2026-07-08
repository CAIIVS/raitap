---
title: "Configuration"
description: "This page describes how to configure the reporting module that generates reports from pipeline outputs."
myst:
  html_meta:
    "description": "This page describes how to configure the reporting module that generates reports from pipeline outputs."
---

```{config-page}
:intro: This page describes how to configure the reporting module that generates reports from pipeline outputs.

:option: use
:allowed: "html", "pdf", null
:default: null
:description: Selects the reporting backend implementation. Set to null to disable reporting.
  The default `reporting=html` config selects `use: html` (`HTMLReporter`);
  use `reporting=pdf` for the PDF renderer.

:option: filename
:allowed: string
:default: "report"
:description: Configured report basename. RAITAP adds the extension for the
  selected reporter, so `HTMLReporter` writes `.html` and `PDFReporter` writes `.pdf`.

:option: sample_selection
:allowed: list of sample IDs, filenames, or zero-based indices; null
:default: null
:description: Optional explicit local-explanation samples to show in the report.
  Strings are matched against dataset sample IDs and filename stems, so
  `ISIC_0024306.jpg` can match `ISIC_0024306`. Integers select zero-based batch
  indices. Invalid, ambiguous, duplicate, or out-of-range entries fail the run
  with a clear error. This affects report rendering only and does not subset the
  data sent through metrics, explainers, visualisers, or tracking.

:option: multirun_report
:allowed: boolean
:default: true
:description: Whether Hydra multiruns should create one merged parent report at
  the sweep directory level. Set to false to keep per-run reports only.

:option: show_original_per_explainer
:allowed: boolean
:default: false
:description: Whether local explanation figures should keep their per-explainer
  original input panels. The default compact layout groups local explanations by
  selected sample, renders one sample thumbnail when possible, and suppresses
  repeated originals when a visualiser supports attribution-only rendering.

:option: show_redundant_robustness_panels
:allowed: boolean
:default: false
:description: Whether empirical robustness report figures should keep duplicate
  clean-input and perturbation-map panels across visualisers. The default
  compact robustness layout renders one figure per selected report sample and
  keeps one canonical owner per facet in the report only; persisted robustness
  PNGs stay self-contained.

:option: include_config
:allowed: boolean
:default: true
:description: Whether to include configuration details in the report.

:option: include_metadata
:allowed: boolean
:default: true
:description: Whether to include metadata (timestamps, versions) in the report.

:option: call.formatting.max_image_width_pt
:allowed: integer, null
:default: null
:description: PDFReporter-only maximum layout width in PDF points for embedded raster figures.
  When null, the usable single-column width on A4 (after margins) is used.

:option: call.formatting.max_image_height_pt
:allowed: integer, null
:default: null
:description: PDFReporter-only maximum layout height in PDF points for embedded raster figures.
  When null, roughly 82% of the inner column height is used to leave room for headings.

:option: call.formatting.figures_max_pages
:allowed: integer, null
:default: null
:description: PDFReporter-only soft cap on how many pages embedded figure sections are allowed to need.
  When set to a positive integer, if a simple estimate exceeds it, width and height limits
  are scaled down so figures take less space per page.

:option: call.formatting.image_raster_multiplier
:allowed: float, null
:default: null (effective default 3.0)
:description: PDFReporter-only pixels per layout point when rasterizing figures (higher = sharper
  in viewers). If set, values below 1.0 are clamped to 1.0.

:option: call.formatting.image_raster_max_edge_px
:allowed: integer, null
:default: null (effective default 2400)
:description: PDFReporter-only maximum longest edge in pixels of the rasterized bitmap after scaling.
  If set, values below 400 are clamped to 400.

:yaml:
reporting:
  use: html
  filename: "experiment_report"
  sample_selection:
    - "ISIC_0024306.jpg"
    - "ISIC_0024372.jpg"
    - 4
  multirun_report: true
  show_original_per_explainer: false
  show_redundant_robustness_panels: false

:cli: reporting=pdf reporting.filename="my_report" reporting.multirun_report=false reporting.show_original_per_explainer=true reporting.show_redundant_robustness_panels=true reporting.call.formatting.figures_max_pages=12

:python:
from raitap.reporting import html

reporting = html(
    filename="experiment_report",
    sample_selection=["ISIC_0024306.jpg", "ISIC_0024372.jpg", 4],
    multirun_report=True,
    show_original_per_explainer=False,
    show_redundant_robustness_panels=False,
)
```
