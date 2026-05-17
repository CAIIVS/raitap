---
title: "Running a quick example"
description: "This page explains how to install RAITAP itself and how to get it running quickly, with a simple demo example. If you want to see how to fully configure your own assessment, skip to the {doc}configuration/index page."
myst:
  html_meta:
    "description": "This page explains how to install RAITAP itself and how to get it running quickly, with a simple demo example. If you want to see how to fully configure your own assessment, skip to the {doc}configuration/index page."
---

# Running a quick example

This page explains how to install RAITAP itself and how to get it running quickly, with a **simple demo example**. If you want to see how to fully configure your own assessment, skip to the {doc}`configuration/index` page.

It is recommended to use `uv`, but `pip` will also work.

## 1. Install RAITAP

First, you need to install the RAITAP package itself.

```{install-tabs}
:uv:
uv add raitap

:pip:
pip install raitap
```

:::{note}
RAITAP supports Python 3.11–3.13. Python 3.14 is not yet
supported (Hydra 1.3.2 limitation). Some underlying libs require older versions (e.g. Marabou < 3.12). RAITAP will handle the interpreter choice for you.
:::

## 2. Run the demo example

RAITAP ships with a self-contained `demo.yaml` you can run with a single flag.
It uses a tiny bundled dataset and CPU execution, so it works out of the box on
any machine.

```{install-tabs}
:uv:
uv run raitap --demo

:pip:
raitap --demo
```

RAITAP does not ship with all the underlying dependencies by default, to avoid massive bloat. This means dependencies required for each specific config must be installed before the run. **RAITAP automatically infers** them from the config.

In some specific setups, you might need to take action:

- If you are using `uv`, it will ask you to run the `uv add` command yourself, or add the `--allow-project-edit` (or `-y`) flag. This is because `uv add` modifies your `pyproject.toml`.
- If you are using `pip` and are not in a virtual environment (`venv`), it will ask to add the `--exec-global` flag. This will modify your global Python setup and is not recommended.

If you wish to manually manage your dependencies, see {doc}`installation`. You can also see a preview of the inferred deps with `--dry-run`.

## 3. Inspect the output

After the run is complete, the `outputs` directory can be found in the directory you ran RAITAP from.

It will contain the run's metadata, the transparency assessment (attributions and visualisations), the robustness assessment (adversarial examples, per-sample verdicts, and an image-pair visualisation), and a PDF report under `reports/`. Refer to the {doc}`understanding-outputs` page for more details.
