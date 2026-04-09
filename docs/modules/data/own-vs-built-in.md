# Using your own data or built-in samples

## Your own data

RAITAP can load your own data by pointing `data.source` to a local file or
directory.

Supported inputs include:

- A directory of images such as `.jpg`, `.png`, `.bmp`, or `.webp`
- A single image file
- A CSV, TSV, or Parquet file
- A directory containing CSV, TSV, or Parquet files

Example:

```yaml
data:
  source: "./data/images" # a directory of images
  labels:
    source: "./data/labels.csv"
    id_column: "image"
    column: "label"
```

If you want to evaluate metrics against ground-truth labels, configure the
optional `data.labels` block as described in [Configuration](configuration.md).

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
