# Reporting

The reporting module generates compact reports from pipeline
outputs. Reports are assembled semantically from metrics, typed transparency
summaries, selected local explanations, and robustness evidence instead of
embedding every PNG artifact written by a run.

The default `HTMLReporter` renders a browser-friendly HTML report and linked
CSS. The legacy borb `PDFReporter` remains available through
`reporting=pdf_borb` when PDF output is required.

```{toctree}
:maxdepth: 1
:caption: Reporting module documentation

configuration
output
```
