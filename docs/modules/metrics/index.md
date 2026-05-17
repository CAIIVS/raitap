---
title: "Metrics"
description: "The metrics module scores model predictions against available targets and produces aggregated evaluation results."
myst:
  html_meta:
    "description": "The metrics module scores model predictions against available targets and produces aggregated evaluation results."
---

# Metrics

The metrics module scores model predictions against available targets and
produces aggregated evaluation results.

## Providing ground-truth labels

Without labels the pipeline falls back to **using model predictions as their
own targets**, which trivially scores 100% and is not informative. Configure
`data.labels` to point at the ground-truth labels alongside your samples; see
[Data configuration](../data/configuration.md) for the available
`labels.source`, `labels.column`, `labels.id_column`, and `labels.encoding`
options.

When labels are missing or fail to align with predictions, raitap emits a
warning so the resulting metrics are clearly flagged as fallback values.

```{toctree}
:maxdepth: 1
:caption: Metrics module documentation

configuration
frameworks-and-libraries
output
```
