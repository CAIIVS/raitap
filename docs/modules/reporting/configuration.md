```{config-page}
:intro: This page describes how to configure the reporting module that generates PDF reports from pipeline outputs.

:option: _target_
:allowed: "PDFReporter", null
:default: null
:description: Hydra target for the reporting backend implementation. Set to null to disable reporting.

:option: filename
:allowed: string
:default: "report.pdf"
:description: Name of the generated PDF report file.

:option: multirun_report
:allowed: boolean
:default: true
:description: Whether Hydra multiruns should create one merged parent report at
  the sweep directory level. Set to false to keep per-run reports only.

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
:description: Maximum layout width in PDF points for embedded raster figures (e.g. PNGs).
  When null, the usable single-column width on A4 (after margins) is used.

:option: formatting.max_image_height_pt
:allowed: integer, null
:default: null
:description: Maximum layout height in PDF points for embedded raster figures (e.g. PNGs).
  When null, roughly 82% of the inner column height is used to leave room for headings.

:option: formatting.figures_max_pages
:allowed: integer, null
:default: null
:description: Soft cap on how many pages embedded figure sections are allowed to need.
  When set to a positive integer, if a simple estimate exceeds it, width and height limits
  are scaled down so figures take less space per page.

:option: formatting.image_raster_multiplier
:allowed: float, null
:default: null (effective default 3.0)
:description: Pixels per layout point when rasterizing figures for borb (higher = sharper
  in viewers). If set, values below 1.0 are clamped to 1.0.

:option: formatting.image_raster_max_edge_px
:allowed: integer, null
:default: null (effective default 2400)
:description: Maximum longest edge in pixels of the rasterized bitmap after scaling.
  If set, values below 400 are clamped to 400.

:yaml:
reporting:
  _target_: "PDFReporter"
  filename: "experiment_report.pdf"
  multirun_report: true
  formatting:
    figures_max_pages: 10

:cli: reporting=pdf reporting.filename="my_report.pdf" reporting.multirun_report=false reporting.formatting.figures_max_pages=10
```
