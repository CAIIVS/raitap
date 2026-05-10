```{config-page}
:intro: This page describes how to configure the reporting module that generates reports from pipeline outputs.

:option: _target_
:allowed: "raitap.reporting.HTMLReporter", "raitap.reporting.PDFReporter", null
:default: null
:description: Hydra target for the reporting backend implementation. Set to null to disable reporting.
  The default `reporting=html` config selects `raitap.reporting.HTMLReporter`;
  use `reporting=pdf` for the legacy borb PDF renderer.

:option: filename
:allowed: string
:default: "report.pdf"
:description: Configured report filename. `HTMLReporter` uses the configured basename with
  a `.html` suffix, while the legacy borb `PDFReporter` uses this value unchanged.

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

:option: formatting.max_image_width_pt
:allowed: integer, null
:default: null
:description: Legacy borb-only maximum layout width in PDF points for embedded raster figures.
  When null, the usable single-column width on A4 (after margins) is used.

:option: formatting.max_image_height_pt
:allowed: integer, null
:default: null
:description: Legacy borb-only maximum layout height in PDF points for embedded raster figures.
  When null, roughly 82% of the inner column height is used to leave room for headings.

:option: formatting.figures_max_pages
:allowed: integer, null
:default: null
:description: Legacy borb-only soft cap on how many pages embedded figure sections are allowed to need.
  When set to a positive integer, if a simple estimate exceeds it, width and height limits
  are scaled down so figures take less space per page.

:option: formatting.image_raster_multiplier
:allowed: float, null
:default: null (effective default 3.0)
:description: Legacy borb-only pixels per layout point when rasterizing figures (higher = sharper
  in viewers). If set, values below 1.0 are clamped to 1.0.

:option: formatting.image_raster_max_edge_px
:allowed: integer, null
:default: null (effective default 2400)
:description: Legacy borb-only maximum longest edge in pixels of the rasterized bitmap after scaling.
  If set, values below 400 are clamped to 400.

:yaml:
reporting:
  _target_: "raitap.reporting.HTMLReporter"
  filename: "experiment_report.pdf"
  multirun_report: true
  show_original_per_explainer: false
  show_redundant_robustness_panels: false

:cli: reporting=pdf reporting.filename="my_report.pdf" reporting.multirun_report=false reporting.show_original_per_explainer=true reporting.show_redundant_robustness_panels=true
```
