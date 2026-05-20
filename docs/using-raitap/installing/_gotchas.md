<!-- Shared partial. Included by automatic.md and manual.md via {include}.
     Do NOT add it to a toctree and do NOT put (label)= anchors here:
     it is rendered into two pages, so anchors would be defined twice. -->
:::{important}
Replace `BACKEND` in the snippets below with `torch` or `onnx` depending on your setup.
:::

:::::{tab-set}

::::{tab-item} CPU
On **Linux**, the plain `torch` package from PyPI is the CUDA build. Route it to the CPU index by adding the following to your `pyproject.toml`:

```toml
[project]
dependencies = [
  "raitap[BACKEND-cpu]>=0.9",
  "torch;       sys_platform == 'linux'", # redeclare to make the index below apply
  "torchvision; sys_platform == 'linux'", # redeclare to make the index below apply
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }
```

::::

::::{tab-item} CUDA
`torch` does not publish separate CUDA builds on PyPI. You control the version via the index by adding the following to your `pyproject.toml`:

```toml
[project]
dependencies = [
  "raitap[BACKEND-cuda]>=0.9; sys_platform != 'darwin'",
  "torch;       sys_platform != 'darwin'", # redeclare to make the index below apply
  "torchvision; sys_platform != 'darwin'", # redeclare to make the index below apply
]

[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu126" # CUDA 12.6, modern default; for older GPUs see below
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cuda" }
torchvision = { index = "pytorch-cuda" }
```

:::{dropdown} Builds for older NVIDIA GPUs (Volta / V100)
The `cu126` wheels may not suit older cards.

1. Find your GPU generation on NVIDIA's [current](https://developer.nvidia.com/cuda/gpus)
/ [legacy](https://developer.nvidia.com/cuda/gpus/legacy) pages and a matching
release on the [PyTorch releases page](https://github.com/pytorch/pytorch/releases).
1. Set the index `url` in your `pyproject.toml` to the matching family (see above), then run `uv lock`.
2. Run RAITAP.
:::
::::

::::{tab-item} Intel GPU
`triton-xpu`'s PyPI releases are either yanked or pre-release. Route it to the Intel index by adding the following to your `pyproject.toml`:

```toml
[project]
dependencies = [
  "raitap[BACKEND-intel]>=0.9; sys_platform != 'darwin'",
  "torch;       sys_platform != 'darwin'", # redeclare to make the index below apply
  "torchvision; sys_platform != 'darwin'", # redeclare to make the index below apply
  "triton-xpu>=3.0.0rc0; sys_platform != 'darwin' and python_full_version < '3.14'",
]

[[tool.uv.index]]
name = "pytorch-intel"
url = "https://download.pytorch.org/whl/xpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-intel" }
torchvision = { index = "pytorch-intel" }
triton-xpu = { index = "pytorch-intel" }
```

::::

:::::

### Wheel availability

PyTorch indexes don't publish for every Python minor:

| Extra                        | Index        | Python    | Platforms             |
| ---------------------------- | ------------ | --------- | --------------------- |
| `torch-cpu` / `onnx-cpu`     | `/whl/cpu`   | 3.11–3.13 | Linux, macOS, Windows |
| `torch-cuda` / `onnx-cuda`   | `/whl/cu126` | 3.11–3.13 | Linux, Windows        |
| `torch-intel` / `onnx-intel` | `/whl/xpu`   | 3.11–3.13 | Linux, Windows        |

### Faster locking on one OS

If you only target one OS, scope the resolver so it doesn't solve for all
platforms at once:

```toml
[tool.uv]
environments = ["sys_platform == 'linux' and python_version >= '3.12'"]
```
