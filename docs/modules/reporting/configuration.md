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

:option: include_config
:allowed: boolean
:default: true
:description: Whether to include configuration details in the report.

:option: include_metadata
:allowed: boolean
:default: true
:description: Whether to include metadata (timestamps, versions) in the report.

:option: forward_to_tracking
:allowed: boolean
:default: true
:description: Whether to automatically upload the generated PDF to the tracking system (if tracking is enabled).

:yaml:
reporting:
  _target_: "PDFReporter"
  filename: "experiment_report.pdf"
  forward_to_tracking: true

:cli: reporting=pdf reporting.filename="my_report.pdf"
```
