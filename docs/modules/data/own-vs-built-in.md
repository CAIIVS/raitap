# Using your own data or built-in samples

## Your own data

RAITAP can load your own data by pointing `data.source` to a local file or
directory.

Supported inputs include:

- A directory of images such as `.jpg`, `.png`, `.bmp`, or `.webp`
  (walked recursively — nested `ImageFolder`-style layouts are supported)
- A single image file
- A CSV, TSV, or Parquet file
- A directory containing CSV, TSV, or Parquet files

Example (flat directory):

```yaml
data:
  source: "./data/images" # a directory of images
  labels:
    source: "./data/labels.csv"
    id_column: "image"
    column: "label"
```

Example (nested ImageFolder layout — `data/test/<class>/<file>.jpg`):

```yaml
data:
  source: "./data/test"
  labels:
    source: "./data/labels.csv"
    id_column: "image"
    column: "label"
    # id_strategy: "auto"   # default — relative paths auto-detected
```

with `labels.csv` rows like `NORMAL/IM-0001.jpeg,0`. See
{doc}`configuration` for the full `id_strategy` reference.

If you want to evaluate metrics against ground-truth labels, configure the
optional `data.labels` block as described in {doc}`configuration`.

## Built-in samples

RAITAP also includes a few built-in sample datasets that can be referenced by
name through `data.source`.

Available sample names are:

- `imagenet_samples`
- `isic2018`
- `malaria`
- `UdacitySelfDriving`

Example:

```yaml
data:
  source: "imagenet_samples"
```

Built-in samples are useful for quickly testing the pipeline without preparing
your own dataset first. RAITAP downloads them to a local cache automatically
when needed.
