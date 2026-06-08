---
title: "Visualisers"
description: "Each visualiser renders one figure per call. All of them are declared per-explainer in YAML:"
myst:
  html_meta:
    "description": "Each visualiser renders one figure per call. All of them are declared per-explainer in YAML:"
---

# Visualisers

Each visualiser renders one figure per call. All of them are declared per-explainer in YAML:

```{config-tabs}
:yaml:
transparency:
  captum_ig:
    _target_: "CaptumExplainer"
    algorithm: "IntegratedGradients"
    visualisers:
      - _target_: "CaptumImageVisualiser"
      - _target_: "TabularBarChartVisualiser"

:python:
from raitap.transparency import captum, captum_image, tabular_bar_chart

transparency = {
    "captum_ig": captum(
        algorithm="IntegratedGradients",
        visualisers=[captum_image(), tabular_bar_chart()],
    ),
}
```

Visualisers declare which `ExplanationScope`, output space, and method families they support;
the factory rejects mismatches at YAML parse time. See
{doc}`frameworks-and-libraries` for the at-a-glance compatibility table and
{doc}`../../contributor/modules/transparency` for the underlying contract.

## Captum

:::::{visualiser-card}
:name: CaptumImageVisualiser
:registry: captum_image
:wraps: `captum.attr.visualization.visualize_image_attr`
:intro: Side-by-side panels — original image on the left, attribution overlay on the right — for each sample in the batch. Use it as the default first-pass figure whenever the explainer produces pixel-level or spatial-map attributions on image inputs.
:how-to-read: Per sample, two panels: the original image and the attribution overlay. Warm regions are pixels that pushed the prediction toward the explained class, cool regions push against it (controlled by `sign`), and brightness is attribution strength — read it as "where the model looked". The `method` kwarg picks the render mode (blended heatmap, bare heatmap, masked image, …).
:kwarg: method
:allowed: blended_heat_map | heat_map | original_image | masked_image | alpha_scaling
:default: `"blended_heat_map"`
:meaning: Captum render mode.
:kwarg: sign
:allowed: all | positive | negative | absolute_value
:default: `"all"`
:meaning: Which contributions to show.
:kwarg: show_colorbar
:default: `True`
:meaning: Whether to add a colorbar next to the attribution panel.
:kwarg: title
:default: `None`
:meaning: Optional attribution panel title forwarded to Captum.
:kwarg: include_original_image
:default: `True`
:meaning: Render the original image next to the attribution panel when `inputs` are available.
:notes: Layer-based methods (`LayerGradCam`, `LayerActivation`, …) emit attribution maps at the chosen layer's spatial resolution (e.g. 7×7 for ResNet-18 `layer4` with 224×224 inputs). When `inputs` are provided, the visualiser bilinearly upsamples such maps to the original image size before rendering so the overlay aligns with the input extent — applied for every `method` except `original_image`. The map still snaps to the layer's cell grid — intrinsic to Grad-CAM, not a visualiser artefact. For tighter localisation, use a shallower layer (e.g. `layer3` → 14×14) or a pixel-space method (`Saliency`, `IntegratedGradients`, `GuidedGradCam`).
:compat: Scope: `LOCAL`. Output spaces: `INPUT_FEATURES`, `IMAGE_SPATIAL_MAP`. Method families: `GRADIENT`, `PERTURBATION`, `SHAPLEY`, `CAM`, `MODEL_AGNOSTIC`, `SURROGATE`. Requires explicit image input metadata (`InputKind.IMAGE`) and an NCHW-compatible attribution shape; rejects tabular, token, and time-series layouts.
:::::

:::::{visualiser-card}
:name: CaptumTimeSeriesVisualiser
:registry: captum_time_series
:wraps: `captum.attr.visualization.visualize_timeseries_attr`
:intro: Overlay of per-channel attribution magnitudes on top of the raw time-series signal. Pick this when the explainer ran on `(T, C)` channels-last inputs and you want to see *when* in the sequence the model focused.
:how-to-read: The x-axis is the sequence position (time); the raw signal is drawn per channel with the per-step attribution magnitude overlaid as colour/intensity, so the bright stretches mark *when* in the sequence the model focused. The `method` kwarg switches between overlaying channels individually, combined, or as a coloured graph.
:kwarg: method
:allowed: overlay_individual | overlay_combined | colored_graph
:default: `"overlay_individual"`
:meaning: Overlay render mode.
:kwarg: sign
:allowed: positive | negative | absolute_value | all
:default: `"absolute_value"`
:meaning: Which contributions to show.
:compat: Scope: `LOCAL`. Output space: `INPUT_FEATURES`. Requires `InputKind.TIME_SERIES` metadata and `(B, T, C)` or `(T, C)` attribution layouts. The `inputs` argument (the original time series) is mandatory — attributions alone are not enough to render the overlay.
:::::

:::::{visualiser-card}
:name: CaptumTextVisualiser
:registry: captum_text
:wraps: a lightweight Matplotlib renderer (Captum's native text visualiser emits HTML rather than a Matplotlib figure)
:intro: Horizontal bar chart of per-token attribution scores. Positive contributions render in warm red, negative in cool blue, so you can read off which tokens pushed the prediction in which direction at a glance.
:how-to-read: One horizontal bar per token, in token order top-to-bottom; bar length is the token's attribution magnitude, warm red means it pushed the prediction up and cool blue means down. Scan it to see which words drove the call.
:kwarg: token_labels
:default: `None`
:meaning: Per-token strings used as y-axis labels. Falls back to `tok_0 … tok_N` when omitted.
:compat: Scope: `LOCAL`. Output space: `TOKEN_SEQUENCE`. Requires text input metadata and a 1-D token attribution tensor with the `TOKENS` / `TOKEN_SEQUENCE` layout.
:::::

## SHAP

:::::{visualiser-card}
:name: ShapBarVisualiser
:registry: shap_bar
:wraps: `shap.summary_plot(plot_type="bar")`
:intro: Mean-absolute-attribution bar chart across the selected batch, one bar per input feature. Use it as the headline "what matters on average?" figure for any tabular or interpretable-features explanation.
:how-to-read: One bar per feature, length is the mean absolute SHAP value across the batch — the sign-agnostic "how much does this feature matter on average". Bars are sorted by importance and the top `max_display` are shown; it answers which features, not which direction.
:kwarg: feature_names
:default: `None`
:meaning: Optional list of feature labels. Falls back to SHAP's `f0 … fN` defaults.
:kwarg: max_display
:default: `20`
:meaning: Maximum number of features to render.
:compat: Scope: `LOCAL` (consumes local attributions). Produces an aggregated visual summary so reporting places the figure under aggregated explanations. Output spaces: `INPUT_FEATURES`, `INTERPRETABLE_FEATURES`. Method family: `SHAPLEY`. Requires `(B, F)` tabular or interpretable attributions.
:::::

:::::{visualiser-card}
:name: ShapBeeswarmVisualiser
:registry: shap_beeswarm
:wraps: `shap.summary_plot()` (default `plot_type="dot"`)
:intro: SHAP beeswarm — one dot per sample per feature, coloured by the feature value, positioned by the SHAP score. Reach for it when the bar chart hides distributional information you care about (e.g. "income matters, but only when it is high").
:how-to-read: One row per feature (most important on top), one dot per sample placed by its SHAP score on the x-axis (left pushes the output down, right up) and coloured by the feature's value (red high, blue low). Horizontal spread shows the distribution of effects; a red→blue gradient along x tells you whether high feature values push the prediction up or down.
:kwarg: feature_names
:default: `None`
:meaning: Optional list of feature labels.
:kwarg: max_display
:default: `20`
:meaning: Maximum number of features to render.
:compat: Scope: `LOCAL` consumed, aggregated visual summary produced. Output spaces: `INPUT_FEATURES`, `INTERPRETABLE_FEATURES`. Method family: `SHAPLEY`. Requires `(B, F)` tabular or interpretable attributions; passing `inputs` unlocks the colour-by-feature-value channel.
:::::

:::::{visualiser-card}
:name: ShapWaterfallVisualiser
:registry: shap_waterfall
:wraps: `shap.plots.waterfall`
:intro: Per-sample waterfall chart: starts at the baseline `expected_value` and walks through each feature's contribution to reach the model output. Pick a single sample with `sample_index` when you want to explain a specific prediction end-to-end.
:how-to-read: For one sample, bars start at the baseline `expected_value` at the bottom and stack each feature's signed contribution (red pushes the output up, blue down) to land on the final model output at the top. Read it bottom-to-top as the step-by-step account of one prediction.
:kwarg: feature_names
:default: `None`
:meaning: Optional list of feature labels.
:kwarg: expected_value
:default: `0.0`
:meaning: Model baseline (`explainer.expected_value` in SHAP).
:kwarg: sample_index
:default: `0`
:meaning: Which row of the batch to render.
:kwarg: max_display
:default: `10`
:meaning: Maximum number of features to show before grouping the rest.
:compat: Scope: `LOCAL`. Requires `(B, F)` tabular or interpretable attributions.
:::::

:::::{visualiser-card}
:name: ShapForceVisualiser
:registry: shap_force
:wraps: `shap.plots.force(matplotlib=True)`
:intro: Per-sample force plot showing positive (red) and negative (blue) feature pushes around the baseline. Compact alternative to the waterfall when you need many local explanations side by side rather than one detailed breakdown.
:how-to-read: For one sample, features push along a horizontal axis around the baseline: red features push the output higher, blue lower, and bar width is the size of each contribution. Where the opposing pushes meet is the final prediction.
:kwarg: feature_names
:default: `None`
:meaning: Optional list of feature labels.
:kwarg: expected_value
:default: `0.0`
:meaning: Model baseline / SHAP base value.
:kwarg: sample_index
:default: `0`
:meaning: Which row of the batch to render.
:compat: Scope: `LOCAL`. Requires `(B, F)` tabular or interpretable attributions. Saved as a PNG via Matplotlib (RAITAP does not use SHAP's HTML force-plot backend).
:::::

:::::{visualiser-card}
:name: ShapImageVisualiser
:registry: shap_image
:wraps: a custom RAITAP Matplotlib renderer (SHAP's native `shap.image_plot` is not used; this keeps the layout consistent with the other RAITAP image visualisers and adds sample-aware titles and colorbar control)
:intro: Paired-panel renderer for pixel-level SHAP values: original image on the left, channel-summed heatmap overlay on the right. Restricted to `GradientExplainer` and `DeepExplainer` — the only SHAP explainers that produce meaningful per-pixel scores.
:how-to-read: Per sample, the original image and a channel-summed SHAP heatmap. Red pixels raise the explained class's score, blue pixels lower it, and intensity is the magnitude — "which pixels move this class, and in which direction".
:kwarg: max_samples
:default: `4`
:meaning: Maximum number of images displayed side by side.
:kwarg: title
:default: `None`
:meaning: Optional attribution-panel title (falls back to algorithm name).
:kwarg: include_original_image
:default: `True`
:meaning: Render the original image next to the heatmap.
:kwarg: show_colorbar
:default: `True`
:meaning: Add a SHAP colorbar in the paired layout.
:kwarg: cmap
:default: `"coolwarm"`
:meaning: Matplotlib colormap for the heatmap overlay.
:kwarg: overlay_alpha
:default: `0.65`
:meaning: Alpha for the SHAP heatmap overlay.
:compat: Scope: `LOCAL`. Output space: `INPUT_FEATURES`. Method family: `GRADIENT`. Requires explicit image input metadata and `(B, C, H, W)` attributions; refuses explanations from non-pixel SHAP explainers such as `KernelExplainer` or `TreeExplainer`.
:::::

## Generic

:::::{visualiser-card}
:name: LayerActivationVisualiser
:registry: layer_activation
:intro: Renders layer-space attributions from captum `Layer*` methods (LayerConductance, LayerActivation, LayerIntegratedGradients, ...) that align to a hidden layer rather than the input. Use it when you attribute to an internal layer and want a magnitude summary instead of an input-space heatmap.
:how-to-read: One panel per sample. A convolutional layer `(C, H, W)` shows one bar per channel (mean absolute attribution); a linear layer `(F,)` shows one bar per feature; a sequence layer `(tokens, hidden)` (e.g. a ViT block) shows a magnitude heatmap. Magnitudes only. These are layer attributions, not input-aligned, so do not read them as a pixel heatmap.
:compat: Scope: `LOCAL`. Output spaces: `LAYER_ACTIVATION` only. Payload kind: `ATTRIBUTIONS`. Pair it with a `Layer*` captum explainer that sets `layer_path`.
:kwarg: max_samples
:default: `8`
:meaning: Maximum number of samples to render (one panel each); larger batches are truncated.

```yaml
transparency:
  my_layer_conductance:
    _target_: CaptumExplainer
    algorithm: LayerConductance
    constructor:
      layer_path: layer4.2.conv3   # dotted path to the target layer
    call:
      target: 0
    visualisers:
      - _target_: LayerActivationVisualiser
```
:::::

:::::{visualiser-card}
:name: StructuredPayloadSummaryVisualiser
:registry: structured_payload_summary
:intro: Renders additive structured payloads (Captum convergence deltas, SHAP base values) that an explainer emits alongside the principal attribution. Use it to surface the per-sample diagnostics persisted under `payloads/` as a quick bar summary.
:how-to-read: One bar panel per matching structured payload, one bar per sample along the x-axis with the payload value on the y-axis. For a `convergence_delta` panel, bars near zero mean the attribution sum closely matched the completeness target for that sample; larger magnitudes flag samples where the approximation drifted.
:compat: Scope: `LOCAL`. Payload kind: `ATTRIBUTIONS`. Structured payload kinds: `convergence_delta`, `base_value`. Reads `VisualisationContext.structured_payloads`; renders only payloads whose kind it declares. Any-of: it is compatible as long as the explanation carries at least one declared kind (an IG run supplies `convergence_delta`, a modern SHAP run supplies `base_value`), so attach it to an explainer that emits a structured payload.

```yaml
transparency:
  ig:
    _target_: CaptumExplainer
    algorithm: IntegratedGradients
    call:
      target: 0
      return_convergence_delta: true   # emits the convergence_delta payload
    visualisers:
      - _target_: StructuredPayloadSummaryVisualiser
```
:::::

:::::{visualiser-card}
:name: TabularBarChartVisualiser
:registry: tabular_bar_chart
:wraps: a small Matplotlib renderer (no third-party plotting dependency)
:intro: Framework-agnostic mean-absolute-attribution bar chart for tabular features. Use it when the explainer is Captum (or anything else) rather than SHAP and you still want the same "what matters on average?" aggregated summary.
:how-to-read: One bar per feature, length is the mean absolute attribution across the batch — the framework-agnostic "what matters on average" for non-SHAP explainers, sorted by importance. Magnitude only, no direction.
:kwarg: feature_names
:default: `None`
:meaning: List of feature names for x-axis labels.
:compat: Scope: `LOCAL` consumed, aggregated visual summary produced. Output spaces: `INPUT_FEATURES`, `INTERPRETABLE_FEATURES`. All method families are accepted. Requires `(B, F)` tabular attributions; rejects image, text, and time-series modalities.
:::::

## Detection

:::::{visualiser-card}
:name: DetectionImageVisualiser
:registry: detection_image
:intro: Renders one figure per detected box for any backend whose `task_kind == detection` (torchvision Faster R-CNN / RetinaNet / SSD). Each figure shows the original image with the reference bounding box outlined and the per-pixel attribution heatmap overlaid.
:how-to-read: One figure per detected box: the original image with that detection's reference box outlined and the per-pixel attribution heatmap overlaid — warm pixels are the evidence supporting *that* box. The title carries the label name (or `class N`), the detection score, and the `display/raw` box index pair for provenance.
:kwarg: method
:allowed: blended_heat_map | heat_map | masked_image | alpha_scaling
:default: `None` → renderer default (`blended_heat_map`)
:meaning: Overlay render mode. Honoured only for captum-sourced detections; a set value the resolved renderer can't honour emits a `UserWarning`.
:kwarg: sign
:allowed: all | positive | negative | absolute_value
:default: `None` → family-auto (`positive` for CAM, else `all`)
:meaning: Which contributions to show. A set value overrides the family-auto sign; an unsupported value emits a `UserWarning`.
:kwarg: show_colorbar
:default: `None` → colorbar shown
:meaning: Gates the attribution colorbar (renderer-agnostic). Set `false` to suppress it.
:kwarg: title
:default: `None` → report group falls back to `ClassName_index`
:meaning: Sets the report group name for this visualiser's figures (covers issue #225's detection half). Does not change the per-box figure title (label + score + box index).
:notes: Compatible with all attribution method families that produce per-pixel maps (gradient, perturbation, shapley, cam, model-agnostic, surrogate).

```yaml
transparency:
  my_ig_explainer:
    _target_: CaptumExplainer
    algorithm: IntegratedGradients
    call:
      target: 0                  # required — wrapper exposes one scalar channel
    raitap:
      detection:
        score_threshold: 0.5     # default; drop detections below this
        max_boxes: 5             # default; cap K per sample
        iou_threshold: 0.5       # default; used by reference_match target
    visualisers:
      - _target_: DetectionImageVisualiser
        # all optional; omit for the default figure. Allowed values per kwarg
        # are in the table above.
        method: blended_heat_map
        sign: positive
        show_colorbar: true
        title: "Integrated Gradients"
```

The pipeline emits one `ExplanationResult` per detected box (top-K after threshold filtering), each carrying a `DetectionBox` with the reference xyxy / score / label. Results from the same sample share `original_sample_index` so reporting groups them visually via the sample-id chip.
:compat: Scope: `LOCAL`. Output space: `DETECTION_BOXES`. Supported task: `detection`. Requires `VisualisationContext.detection_box` to be set (populated automatically by the detection explain phase).
:::::

`method` / `sign` are honoured only when the resolved renderer supports them —
the renderer is chosen automatically from the attribution's source library, not
configured directly. The captum renderer honours both; the house renderer
honours only `sign` `all`/`positive`; the SHAP renderer honours neither. When
you set a `method`/`sign` the resolved renderer can't honour, raitap emits a
`UserWarning` naming the field and source library instead of silently ignoring
it. `show_colorbar` is renderer-agnostic (it gates the figure's attribution
colorbar directly), so it applies regardless of source library.

:::{note}
**Adding a renderer (contributors).** `visualise()` forwards style kwargs to the
resolved renderer's `draw(**style)`. Any renderer registered via
`@image_renderer` **must** accept `**style` (the `ImageAttributionRenderer`
protocol declares it); a renderer that omits it raises `TypeError` once a user
sets `method`. To participate in the unhonoured-field warning, declare the
optional `honours_method` (bool) and `honoured_signs` (`frozenset[str]`) class
constants — they are read via `getattr` with honour-all defaults, so they are
not required.
:::

## Reporting helpers

:::::{visualiser-card}
:name: InputThumbnailVisualiser
:registry: input_thumbnail
:intro: Compact preview of the original input, used by the report builder to render one shared sample thumbnail in sample-header rows (not a typical user-configured visualiser). Image inputs only; falls back gracefully when no compatible input is available.
:how-to-read: Just a thumbnail of the original input — no attribution overlay. The report uses it to anchor each sample row visually so the explanation figures below it have a reference image.
:kwarg: title
:default: `"Input"`
:meaning: Caption shown above the thumbnail.
:kwarg: max_samples
:default: `8`
:meaning: Maximum number of thumbnails rendered side by side.
:compat: Scope: `LOCAL`. Requires image input metadata (`InputKind.IMAGE` or NCHW layout) and a non-`None` `inputs` tensor.
:::::
