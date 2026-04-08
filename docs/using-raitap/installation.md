# Installation

## 1. Install RAITAP

```{install-tabs}
:uv: uv pip install raitap
:pip: pip install raitap
```

## 2. Install dependencies

### Execution dependencies

RAITAP supports both PyTorch and ONNX models, and both CPU and GPU execution. Hence, you need to install the dependencies for your specific setup.

Here is an example command for PyTorch on CPU with ONNX support:

```{install-tabs}
:uv: uv sync --extra onnx-cpu
:pip: pip install "raitap[onnx-cpu]"
```

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

```{install-tabs}
:uv: uv sync --extra transparency
:pip: pip install "raitap[transparency]"
```

If you know you will use a single underlying framework (here Captum), you can run the following instead:

```{install-tabs}
:uv: uv sync --extra captum
:pip: pip install "raitap[captum]"
```

### Single one-shot command

Of course, optional groups can be combined. For instance:

```{install-tabs}
:uv: uv sync --extra onnx-cpu --extra transparency
:pip: pip install "raitap[onnx-cpu,transparency]"
```
