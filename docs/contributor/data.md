# Contributing to the data module

This page describes how to add new built-in dataset configs to RAITAP.

## Overview

The data module provides built-in sample datasets that can be referenced by name in `data.source`. These configs live in `src/raitap/configs/data/` and define metadata and default loading behavior for common datasets.

## Adding a built-in dataset

To add a new built-in dataset config:

1. **Create the config file**

    Add a YAML config under `src/raitap/configs/data/`:

    ```yaml
    # src/raitap/configs/data/cifar10.yaml
    name: cifar10
    description: CIFAR-10 — 60k 32×32 colour images in 10 classes
    source: cifar10
    ```

2. **Add loading support (optional)**

    If the dataset requires custom loading logic beyond what `load_tensor_from_source()` provides, add it to `src/raitap/data/data.py` or `samples.py`.

3. **Use it**

    ```bash
    uv run raitap data=cifar10
    ```

4. **Update documentation**

    Add the new dataset name to the list in `docs/modules/data/own-vs-built-in.md`.
