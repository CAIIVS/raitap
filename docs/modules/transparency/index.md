# Transparency

The transparency module configures the explainers and visualisers that generate
attributions for your model predictions.

Each `transparency` entry defines one named explainer, its algorithm, and the
visualisers that should render its outputs. The current implementation supports
Captum- and SHAP-based explainers behind the same config surface.

```{toctree}
:maxdepth: 1
:caption: Transparency module documentation

configuration
frameworks-and-libraries
output
```
