# Kitchen-sink example

The example below shows a complete configuration with all top-level modules populated.

If you want to learn how to write such a config, see the {doc}`general` guide.

```yaml
hardware: "gpu"
experiment_name: "My Experiment"

model:
  source: "./models/my-model.onnx"

data:
  name: "my-dataset"
  description: "Internal validation set"
  source: "./data/images"
  labels:
    source: "./data/labels.csv"
    id_column: "image"
    column: "label"
    encoding: "index"

transparency:
  captum_ig:
    _target_: "CaptumExplainer"
    algorithm: "IntegratedGradients"
    constructor: {}
    call:
      target: 0
      baselines:
        source: "./data/baselines"
        n_samples: 8
    visualisers:
      - _target_: "CaptumImageVisualiser"
        constructor:
          method: "blended_heat_map"
          sign: "all"
          show_colorbar: true
          title: "Integrated gradients"
          include_original_image: true
        call:
          max_samples: 4
          show_sample_names: true
  shap_gradient:
    _target_: "ShapExplainer"
    algorithm: "GradientExplainer"
    constructor:
      local_smoothing: 0.0
    call:
      target: 0
      nsamples: 10
      batch_size: 1
      show_progress: true
      progress_desc: "SHAP batches"
      background_data:
        source: "./data/background"
        n_samples: 32
    visualisers:
      - _target_: "ShapImageVisualiser"
        constructor:
          max_samples: 2

metrics:
  _target_: "ClassificationMetrics"
  task: "multiclass"
  num_classes: 7
  num_labels: null
  average: "macro"
  ignore_index: null

tracking:
  _target_: "MLFlowTracker"
  output_forwarding_url: "http://127.0.0.1:5000"
  log_model: false
  open_when_done: true
```
