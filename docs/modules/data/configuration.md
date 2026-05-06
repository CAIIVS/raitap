```{config-page}
:intro: This page describes how to configure the data used to assess the model.

:option: name
:allowed: string
:default: "isic2018"
:description: Name shown in outputs and tracking metadata. 

:option: description
:allowed: string, null
:default: null
:description: Optional human-readable dataset description.

:option: source
:allowed: string, null
:default: null
:description: Path to a local data directory, or a named sample set such as
  `"imagenet_samples"`.

:option: forward_batch_size
:allowed: int, null
:default: null
:description: Optional batch size for the initial model forward pass used to
  produce predictions for metrics, report sample ranking, and `auto_pred`
  explainer targets. If omitted, RAITAP uses its pipeline default of `32`.
  This does not control explainer attribution batching; use
  `transparency.<explainer>.raitap.batch_size` for that.

:option: labels.source
:allowed: string, null
:default: null
:description: Optional path to a labels file. Supported formats are CSV, TSV,
  and Parquet.

:option: labels.id_column
:allowed: string, null
:default: null
:description: Optional sample-ID column used to align labels with filenames,
  for example `"image"`.

:option: labels.column
:allowed: string, null
:default: null
:description: Optional class-label column. If omitted, one-hot numeric columns
  are reduced with `argmax`.

:option: labels.encoding
:allowed: "index", "one_hot", "argmax", null
:default: null
:description: Optional label parsing strategy.

:option: labels.id_strategy
:allowed: "auto", "relative_path", "stem"
:default: "auto"
:description: How label-file ids are matched against discovered sample
  files. `"auto"` (default) inspects the id column and switches to
  `"relative_path"` if any value contains `/` or `\`, otherwise falls back
  to `"stem"`. `"relative_path"` keeps directory components and supports
  nested ImageFolder layouts (e.g. `NORMAL/IM-0001.jpeg`) — required when
  filename stems collide across class subdirs. `"stem"` is the legacy
  flat-dir behaviour.

:yaml:
data:
  name: "my-dataset"
  description: "Internal validation set"
  source: "./data/images"
  forward_batch_size: 32
  labels:
    source: "./data/labels.csv"
    id_column: "image"
    column: "label"
    encoding: "index"
    id_strategy: "auto"

:cli: data.source="./data/images" data.labels.source="./data/labels.csv" data.labels.column=label
```

For nested `ImageFolder`-style layouts (e.g. `data/test/<class>/<file>.jpg`)
see {doc}`own-vs-built-in`.
