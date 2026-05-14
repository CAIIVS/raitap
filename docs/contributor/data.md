# Contributing to the data module

This page describes how to add new built-in sample datasets to RAITAP.

## Overview

RAITAP ships a small set of named sample bundles (e.g. `imagenet_samples`,
`mnist_samples`) that can be referenced by name in `data.source`. They are
**not** Hydra config-group presets — RAITAP no longer ships a `src/raitap/configs/data/`
group. The registration lives in `src/raitap/data/samples.py` as plain
Python dicts:

- `SAMPLE_SOURCES` — name → list of `(url, filename)` pairs to download into
  the per-user cache (`~/.cache/raitap/<name>/`).
- `SAMPLE_LABELS` (optional) — name → `{filename: class_index}` map that gets
  materialised as `labels.csv` alongside the images, so `data.labels.source: <name>`
  resolves automatically.

## Adding a built-in sample dataset

1. **Register the source files** in `src/raitap/data/samples.py`:

    ```python
    SAMPLE_SOURCES["cifar10_samples"] = [
        ("https://…/airplane.png", "airplane.png"),
        ("https://…/cat.png", "cat.png"),
        # …
    ]
    ```

2. **(Optional) Register labels** if the dataset has a known ground truth that
   matches the model's class space:

    ```python
    SAMPLE_LABELS["cifar10_samples"] = {
        "airplane.png": 0,
        "cat.png": 3,
        # …
    }
    ```

3. **Use it** from any consumer config:

    ```yaml
    data:
      name: cifar10_samples
      source: cifar10_samples            # resolves via SAMPLE_SOURCES
      labels:
        source: cifar10_samples          # resolves via SAMPLE_LABELS → labels.csv
        id_column: image
        column: label
    ```

    Or override on the CLI:

    ```bash
    uv run raitap --config-name assessment data.source=cifar10_samples
    ```

4. **Add custom loading logic** in `src/raitap/data/data.py` or `samples.py`
   only if the bundle needs more than the default image/tabular loader.

5. **Update documentation** — add the new sample name to
   `docs/modules/data/own-vs-built-in.md`.

## Sample discovery and label alignment

`data.source` directories are walked **recursively** (`Path.rglob`) and
sample ids are computed as posix-style paths relative to the source root
(e.g. `NORMAL/IM-0001.jpeg`). This supports nested `ImageFolder`-style
layouts where filename stems may collide across class subdirs.

Label alignment is governed by `data.labels.id_strategy`:

- `"auto"` (default) — inspects the id column once. If any value contains
  `/` or `\`, switches to `"relative_path"`; otherwise `"stem"`.
- `"relative_path"` — strips only the file extension during comparison;
  directory components are retained.
- `"stem"` — legacy flat-dir behaviour (`Path(id).stem`).

Both sides (sample ids from disk + label ids from the file) are normalised
the same way before lookup. Duplicate normalised label ids raise a
warning and disable label-based metrics for that run.
