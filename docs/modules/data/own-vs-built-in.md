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
from raitap.api import DataConfig, LabelsConfig

data = DataConfig(
    source="./data/images",  # a directory of images
    labels=LabelsConfig(
        source="./data/labels.csv",
        id_column="image",
        column="label",
    ),
)
```

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
from raitap.api import DataConfig, LabelsConfig

data = DataConfig(
    source="./data/test",
    labels=LabelsConfig(
        source="./data/labels.csv",
        id_column="image",
        column="label",
        # id_strategy="auto",  # default — relative paths auto-detected
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
from raitap.api import DataConfig

data = DataConfig(source="imagenet_samples")
```

Built-in samples are useful for quickly testing the pipeline without preparing
your own dataset first. RAITAP downloads them to a local cache automatically
when needed.
