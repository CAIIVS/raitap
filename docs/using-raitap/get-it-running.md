# Running a quick example

This page shows how to get RAITAP running quickly, with a **simple pre-defined example**. If you want to see how to fully configure your own assessment, skip to the {doc}`configuration/index` page.

:::{note}

This page assumes you have **already installed RAITAP**. If you didn't, see the {doc}`installation` page.
:::

## 1. Install dependencies

Our pre-defined examples uses a PyTorch model and runs a transparency assessment using Captum. Hence, we install the dependencies:

```{install-tabs}
:uv:
uv sync --extra captum --extra torch-cpu

:pip:
pip install "raitap[captum,torch-cpu]"
```

:::{note}

The example is light enough to run on a CPU, hence we used `torch-cpu`. Feel free to use another execution profile (e.g. `torch-cuda`). Refer to {ref}`execution-dependencies` for more details.
:::

## 2. Run the example

Our pre-defined example is the default config shipped with RAITAP. This means you do not need to specify any options to run it.

```{install-tabs}
:uv:
uv run raitap

:pip:
raitap
```

## 3. Inspect the output

After the run is complete, the `outputs` directory can be found in the directory you ran RAITAP from.

It will contain the run's metadata, as well as the transparency assessment (attributions and visualisations). Refer to the {doc}`understanding-outputs` page for more details.
