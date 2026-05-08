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

```{toctree}
:maxdepth: 1
:caption: Robustness module documentation

configuration
frameworks-and-libraries
output
```
