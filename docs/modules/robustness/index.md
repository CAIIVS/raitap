---
title: "Robustness"
description: "The robustness module configures the assessors and visualisers that probe how a model behaves under input perturbations."
myst:
  html_meta:
    "description": "The robustness module configures the assessors and visualisers that probe how a model behaves under input perturbations."
---

# Robustness

The robustness module configures the assessors and visualisers that probe how a
model behaves under input perturbations.

Each `robustness` entry defines one named assessor, its algorithm, and the
visualisers that should render its outputs. The current implementation supports
two complementary methods:

- **Empirical attacks** — try to find an adversarial example within a
  perturbation budget (torchattacks, foolbox).
- **Formal verification** — prove that no adversarial example exists within the
  budget. The module shape already accommodates this; concrete adapters
  (auto_LiRPA, alpha-beta-CROWN) arrive in a follow-up release.

A "non-attack" outcome from an empirical assessor does **not** prove robustness;
it just means the configured attack failed. Use a formal-verification assessor
when you need a robustness proof rather than an attack attempt.

## Providing ground-truth labels

Untargeted attacks need a per-sample reference label to push the model away
from. Without `data.labels`, raitap falls back to `argmax(model(clean))`,
which means the attack only confirms the model's *current* decision is
brittle — not that it disagrees with reality. Configure `data.labels` to
supply real labels; see [Data configuration](../data/configuration.md) for
the `labels.source`, `labels.column`, `labels.id_column`, and
`labels.encoding` options.

When labels are missing, raitap emits a warning so the fallback target is
clearly flagged.

```{toctree}
:maxdepth: 1
:caption: Robustness module documentation

configuration
frameworks-and-libraries
output
visualisers
```
