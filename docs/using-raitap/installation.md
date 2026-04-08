# Installation

## 1. Install RAITAP

::::{tab-set}
:sync-group: install

:::{tab-item} uv
:sync: uv

```shell
uv pip install raitap
```

:::

:::{tab-item} pip
:sync: pip

```shell
pip install raitap
```

:::

::::

## 2. Install dependencies

### Execution dependencies

RAITAP supports both PyTorch and ONNX models, and both CPU and GPU execution. Hence, you need to install the dependencies for your specific setup.

Here is an example command for PyTorch on CPU with ONNX support:

::::{tab-set}
:sync-group: install

:::{tab-item} uv
:sync: uv

```shell
uv sync --extra onnx-cpu
```

:::

:::{tab-item} pip
:sync: pip

```shell
pip install "raitap[onnx-cpu]"
```

:::

::::

Here is the full list of dependency groups:

- `torch-cpu`
- `torch-cuda`
- `torch-intel`
- `onnx-cpu`
- `onnx-cuda`
- `onnx-intel`

:::{note}

- CUDA corresponds to NVIDIA GPUs.
- `torch-intel` uses the Intel XPU API directly.
- `onnx-intel` uses the OpenVINO ONNX Runtime.
- Apple MPS support is coming soon.
:::

### Assessment dependencies

You can then install the dependencies for the assessment modules you want to use.

For instance, if you want to assess the transparency, run:

::::{tab-set}
:sync-group: install

:::{tab-item} uv
:sync: uv

```shell
uv sync --extra transparency
```

:::

:::{tab-item} pip
:sync: pip

```shell
pip install "raitap[transparency]"
```

:::

::::

If you know you will use a single underlying framework (here Captum), you can run the following instead:

::::{tab-set}
:sync-group: install

:::{tab-item} uv
:sync: uv

```shell
uv sync --extra captum
```

:::

:::{tab-item} pip
:sync: pip

```shell
pip install "raitap[captum]"
```

:::

::::

### Single one-shot command

Of course, optional groups can be combined. For instance:

::::{tab-set}
:sync-group: install

:::{tab-item} uv
:sync: uv

```shell
uv sync --extra onnx-cpu --extra transparency
```

:::

:::{tab-item} pip
:sync: pip

```shell
pip install "raitap[onnx-cpu,transparency]"
```

:::

::::
