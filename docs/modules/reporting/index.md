# Reporting

The reporting module generates compact PDF reports from pipeline outputs. Reports
are assembled semantically from metrics, typed transparency summaries, and
selected local explanations instead of embedding every PNG artifact written by a
run.

Rendered visualisations are grouped by `VisualisationResult.scope`; current
section names include Global Explanations, Cohort Explanations, and Local
Explanations.

```{toctree}
:maxdepth: 1
:caption: Reporting module documentation

configuration
output
```
