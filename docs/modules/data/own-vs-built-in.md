---
title: "Using your own data or built-in samples"
description: "RAITAP can load your own data by pointing data.source to a local file or directory."
myst:
  html_meta:
    "description": "RAITAP can load your own data by pointing data.source to a local file or directory."
---

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

```{config-tabs}
:yaml:
data:
  source: "./data/images" # a directory of images
  labels:
    source: "./data/labels.csv"
    id_column: "image"
    column: "label"

:python:
from raitap.data import DataConfig, LabelsConfig

data = DataConfig(
    source="./data/images",  # a directory of images
    labels=LabelsConfig(
        source="./data/labels.csv",
        id_column="image",
        column="label",
    ),
)
```

with `labels.csv` rows like:

```text
image,label
IM-0001.jpeg,0
IM-0002.jpeg,1
```

The ids are bare filenames (no path separators), so `auto` matches by
`stem`. See [How labels match to samples](#how-labels-match-to-samples) below.

Example (nested ImageFolder layout — `data/test/<class>/<file>.jpg`):

```text
data/test/
├── NORMAL/IM-0001.jpeg
├── NORMAL/IM-0002.jpeg
└── PNEUMONIA/IM-0001.jpeg   # colliding stem with NORMAL/
```

```{config-tabs}
:yaml:
data:
  source: "./data/test"
  labels:
    source: "./data/labels.csv"
    id_column: "image"
    column: "label"
    # id_strategy: "auto"   # default — relative paths auto-detected

:python:
from raitap.data import DataConfig, IdStrategy, LabelsConfig

data = DataConfig(
    source="./data/test",
    labels=LabelsConfig(
        source="./data/labels.csv",
        id_column="image",
        column="label",
        # id_strategy=IdStrategy.auto,  # default — relative paths auto-detected
    ),
)
```

with `labels.csv` rows like:

```text
image,label
NORMAL/IM-0001.jpeg,0
NORMAL/IM-0002.jpeg,0
PNEUMONIA/IM-0001.jpeg,1
```

The default `labels.id_strategy: "auto"` detects the path separators and
matches by relative path (extension is stripped during comparison, so
`NORMAL/IM-0001.jpeg` and `NORMAL/IM-0001` both work). Sample order is
sorted by relative posix path. See {doc}`configuration` for the full
`id_strategy` reference.

### How labels match to samples

Labels come from the `data.labels` file, not from folder names. RAITAP does
not infer the class from a parent directory the way torchvision `ImageFolder`
does. Each labels row is tied to a sample two ways:

- **By id** — when `labels.id_column` is set, its value is matched against the
  discovered sample files.
- **By order** — when no `id_column` is set, labels align to samples by sorted
  file order (row 1 to the first file, and so on).

For id matching, `id_strategy` controls how both sides are normalised before
the lookup. The same rule is applied to the sample paths (from disk) and the
label ids (from the file):

| `id_strategy`    | strips          | `NORMAL/IM-0001.jpeg` becomes | use when                                 |
| ---------------- | --------------- | ----------------------------- | ---------------------------------------- |
| `relative_path`  | extension only  | `NORMAL/IM-0001`              | label ids carry the directory (manifest) |
| `stem`           | directory + ext | `IM-0001`                     | flat dir, label ids are bare filenames   |
| `auto` (default) | picks per id    | depends on the id column      | leave it; detects which form fits        |

`auto` switches to `relative_path` as soon as any label id contains `/` or
`\`, otherwise `stem`. The manifest form (`relative_path`/`auto`) is the
conventional one and the safe default. `stem` collapses ids that share a
filename across subdirs (`NORMAL/IM-0001.jpeg` and `PNEUMONIA/IM-0001.jpeg`
both become `IM-0001`), which raises a duplicate-id error.

If you want to evaluate metrics against ground-truth labels, configure the
optional `data.labels` block as described in {doc}`configuration`.

## Built-in samples

RAITAP also includes a few built-in sample datasets that can be referenced by
name through `data.source`.

Available sample names (registered in `src/raitap/data/samples.py`) are:

- `imagenet_samples` — 4 ImageNet test images (tench, shih-tzu, golden retriever, tiger cat), with bundled `labels.csv`.
- `isic2018` — small ISIC 2018 dermoscopy subset (no labels).
- `malaria` — small malaria thin-blood-smear subset (no labels).
- `acas_xu_n1_1` — ACAS Xu N1_1 tabular sample (no labels).
- `UdacitySelfDriving` — small Udacity self-driving subset (no labels).

Example:

```{config-tabs}
:yaml:
data:
  source: "imagenet_samples"

:python:
from raitap.data import DataConfig

data = DataConfig(source="imagenet_samples")
```

Built-in samples are useful for quickly testing the pipeline without preparing
your own dataset first. RAITAP downloads them to a local cache automatically
when needed.
