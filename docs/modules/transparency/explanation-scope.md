# Explanation scope and typed semantics

This page summarizes the user-facing typed semantics used by transparency
artifacts, visualisers, and reports.

## Scope

`ExplanationScope` describes the semantic breadth of an explanation or rendered
visualisation:

| Scope | Meaning |
| --- | --- |
| `LOCAL` | Explains individual input samples. Current Captum and SHAP attribution explainers produce local explanation artifacts. |
| `COHORT` | Summarizes the selected input batch or cohort. Current SHAP bar, SHAP beeswarm, and tabular bar visualisers produce cohort visual summaries when they aggregate local attributions. |
| `GLOBAL` | Represents a dataset, population, or model-wide result. The enum keeps this report concept available, but current built-in visualisers do not promote arbitrary batches to global outputs. |

The `COHORT` distinction is intentional. A SHAP plotting API may call a bar or
beeswarm figure "global", but RAITAP only treats it as global when a future
first-class dataset, population, or model-level contract proves that scope.

## Scope definition step

`ScopeDefinitionStep` records where the scope was defined:

| Step | Meaning |
| --- | --- |
| `EXPLAINER_OUTPUT` | The explainer produced an artifact with this scope. |
| `VISUALISER_SUMMARY` | The visualiser changed the result scope by summarizing another explanation artifact. |

For example, an attribution explainer can produce local attributions with
`EXPLAINER_OUTPUT`. A summary visualiser can consume those local attributions and
produce a cohort figure with `VISUALISER_SUMMARY`.

## Semantics carried by results

`ExplanationResult.semantics` describes the computed explanation artifact. It is
a typed contract, not a narrative description. The contract records the artifact
scope, scope definition step, payload kind, method families, target, sample
selection, input metadata, and output-space metadata.

`VisualisationResult.scope` describes what the rendered figure represents. A
visualiser that preserves the explanation scope leaves the scope unchanged. A
visualiser that summarizes a collection declares the produced scope explicitly.

## Sample selection

RAITAP separates stable sample identity from display labels:

| Field | Purpose |
| --- | --- |
| `sample_ids` | Stable IDs from the data pipeline, when available. |
| `sample_display_names` | Optional labels used for plot titles. |

Display names are not stable identity. RAITAP must not infer dataset,
population, or global semantics from sample names shown in plots.

## Output space

`ExplanationOutputSpace` describes what attribution values are aligned to:

| Output space | Typical use |
| --- | --- |
| `INPUT_FEATURES` | Attributions aligned with input features, pixels, or tabular columns. |
| `INTERPRETABLE_FEATURES` | Attributions aligned with an interpretable feature representation. |
| `LAYER_ACTIVATION` | Attributions aligned with an internal model layer. |
| `IMAGE_SPATIAL_MAP` | CAM-style or spatial image maps that may need interpolation. |
| `TOKEN_SEQUENCE` | Token-level text attributions. |

Output-space inference relies on explicit input metadata and algorithm semantics.
Shape alone is not enough to decide whether a tensor is tabular, token, image,
or time-series data.

## Reporting placement

Reports use typed visualisation scope rather than legacy report-placement
strings. Section order is:

1. Metrics
2. Global Explanations
3. Cohort Explanations
4. Local Explanations

Empty sections are omitted. With current built-ins, SHAP bar, SHAP beeswarm, and
tabular bar summaries belong in `Cohort Explanations`, while per-sample image,
text, and time-series visualisations belong in `Local Explanations`.
