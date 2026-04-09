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

:yaml:
data:
  name: "my-dataset"
  description: "Internal validation set"
  source: "./data/images"
  labels:
    source: "./data/labels.csv"
    id_column: "image"
    column: "label"
    encoding: "index"

:cli: data.source="./data/images" data.labels.source="./data/labels.csv" data.labels.column=label
```
