---
title: "Running a quick example"
description: "This page explains how to install RAITAP itself and how to get it running quickly, with a simple demo example. If you want to see how to fully configure your own assessment, skip to the {doc}configuration/index page."
myst:
  html_meta:
    "description": "This page explains how to install RAITAP itself and how to get it running quickly, with a simple demo example. If you want to see how to fully configure your own assessment, skip to the {doc}configuration/index page."
---

# Running a quick example

This page explains how to install RAITAP itself and how to get it running quickly, with a **simple demo example**. If you want to see how to fully configure your own assessment, skip to the {doc}`installing/index` and {doc}`configuration/index` pages.

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

RAITAP does not ship with all the underlying dependencies by default, to avoid massive bloat. The required dependencies are automatically inferred and installed by default.

In some specific setups, you might need to take action for the automatic install to occur:

- On `uv`, RAITAP may ask for <a href="flags.html#flag-allow-project-edit"><code>--allow-project-edit</code></a> (or `-y`).
- On `pip` without a venv, RAITAP may ask for <a href="flags.html#flag-exec-global"><code>--exec-global</code></a>.

## 3. Inspect the output

After the run is complete, the `outputs` directory can be found in the directory you ran RAITAP from.

It will contain the run's metadata, the transparency assessment (attributions and visualisations), the robustness assessment (adversarial examples, per-sample verdicts, and an image-pair visualisation), and a PDF report under `reports/`. Refer to the {doc}`understanding-outputs` page for more details.
