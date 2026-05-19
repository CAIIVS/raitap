---
title: "Consuming RAITAP from another project"
description: "How to depend on RAITAP from a downstream pyproject.toml. Covers the [tool.uv.sources] propagation gap, the cross-platform extras pattern, and the resolver knobs needed for torch-cuda / torch-intel wheels."
myst:
  html_meta:
    "description": "How to depend on RAITAP from a downstream pyproject.toml. Covers the [tool.uv.sources] propagation gap, the cross-platform extras pattern, and the resolver knobs needed for torch-cuda / torch-intel wheels."
---

(consuming-raitap)=

# Consuming RAITAP from another project

When you `pip install raitap` (or `uv add raitap`) into another project's
`pyproject.toml`, `uv` discards RAITAP's `[tool.uv.sources]` and
`[[tool.uv.index]]` blocks — those settings only apply to the project that
owns the `pyproject.toml`. Without re-declaring them in your project,
`torch-cuda` / `torch-intel` extras resolve from PyPI, which either fails
("no wheel for current platform") or installs the wrong build.

This page lists the boilerplate a downstream project needs.

(downstream-cross-platform-extras)=

## Cross-platform extras pattern

Pin RAITAP once per platform so each OS pulls the right backend extra. The
`sys_platform` markers keep `uv` from satisfying all splits at the same
time:

```toml
# pyproject.toml of the *consuming* project
[project]
dependencies = [
  "raitap[transparency,torch-cuda]>=0.9; sys_platform == 'linux'",
  "raitap[transparency,torch-intel]>=0.9; sys_platform == 'win32'",
  "raitap[transparency,torch-cpu]>=0.9;  sys_platform == 'darwin'",
]
```

Pick the assessment extras (`transparency`, `robustness`, `metrics`,
`tracking`, `reporting`, individual library names like `captum` / `mlflow`)
the same way you would in a flat install — see {doc}`installation` for the
full list.

(downstream-uv-sources)=

## Re-declare RAITAP's index routing

Copy the relevant subset of RAITAP's own routing. Both blocks are
mandatory: `[[tool.uv.index]]` declares the PyTorch wheel indexes,
`[tool.uv.sources]` tells `uv` which extras pull from which index.

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[[tool.uv.index]]
name = "pytorch-intel"
url = "https://download.pytorch.org/whl/xpu"
explicit = true

[tool.uv.sources]
torch = [
  { index = "pytorch-cpu",   extra = "torch-cpu"   },
  { index = "pytorch-cuda",  extra = "torch-cuda"  },
  { index = "pytorch-intel", extra = "torch-intel" },
]
torchvision = [
  { index = "pytorch-cpu",   extra = "torch-cpu"   },
  { index = "pytorch-cuda",  extra = "torch-cuda"  },
  { index = "pytorch-intel", extra = "torch-intel" },
]
# triton-xpu's only PyPI release (0.0.2) is yanked. The 3.x wheels live on
# the pytorch-intel index; without this entry uv resolves to a Linux-only
# pre-release from PyPI on Windows.
triton-xpu = [
  { index = "pytorch-intel", extra = "torch-intel" },
]
```

Drop the indexes / source entries you don't use. If your project only
targets Linux + CUDA, you only need `pytorch-cuda` plus the `torch` /
`torchvision` source entries scoped to `torch-cuda`.

You also need to **re-declare the platform-specific packages as direct
dependencies** so the sources actually bind. Without a direct entry for
`torch`, `torchvision`, and (for Intel) `triton-xpu`, `uv` treats them as
transitive and skips the source mapping:

```toml
[project]
dependencies = [
  "raitap[transparency,torch-cuda]>=0.9; sys_platform == 'linux'",
  "torch;        sys_platform == 'linux'",
  "torchvision;  sys_platform == 'linux'",

  "raitap[transparency,torch-intel]>=0.9; sys_platform == 'win32'",
  "torch;       sys_platform == 'win32'",
  "torchvision; sys_platform == 'win32'",
  "triton-xpu>=3.0.0; sys_platform == 'win32' and python_full_version < '3.14'",
]
```

If you also depend on the OpenVINO ONNX runtime, add `onnxruntime-openvino`
and `openvino` to the `'win32'` / `'linux'` blocks alongside `torch-intel`.

(downstream-prerelease)=

## `prerelease = "allow"` for `torch-intel`

Even with the source mapping above, `uv` reads PyPI's `triton-xpu`
metadata, which contains pre-release versions, and refuses to use them
under the default `prerelease = "if-necessary-or-explicit"`. Add:

```toml
[tool.uv]
prerelease = "allow"
```

You can scope this to a single dependency-group / environment if you don't
want the relaxation to apply globally.

(downstream-wheel-matrix)=

## Python × extra wheel matrix

The PyTorch wheel indexes don't publish for every Python minor. The table
below summarises which `(extra, Python, platform)` combinations have
wheels today.

| Extra | `pytorch-*` index | Python minors | Platforms | Notes |
|---|---|---|---|---|
| `torch-cpu` | `pytorch-cpu` | 3.11–3.13 | Linux, macOS, Windows | Always available; safe fallback. |
| `torch-cuda` | `pytorch-cuda` (`/whl/cu126`) | 3.11–3.13 | Linux, Windows | CUDA 12.6 wheels; older CUDA needs a different index (see {ref}`execution-dependencies`). |
| `torch-intel` | `pytorch-intel` (`/whl/xpu`) | 3.11–3.13 | Linux, Windows | No macOS wheels. Requires `triton-xpu` routed via `[tool.uv.sources]`. |
| `onnx-cpu` / `onnx-cuda` / `onnx-intel` | inherits the corresponding `torch-*` index | 3.11–3.13 | Same as the matching `torch-*` row | Bundles `torch` + `torchvision` until ONNX no longer needs them internally. |

RAITAP itself supports Python 3.11–3.13 (Hydra 1.3.2 blocks 3.14 until
upstream catches up). When the resolver fails with a "no wheel for current
platform" error, cross-reference this table before changing extras —
usually the answer is a Python version that the upstream index actually
publishes for.

(downstream-environments)=

## Scoping the resolver with `tool.uv.environments`

A consuming project that only ever runs on one OS doesn't need `uv` to
satisfy every `sys_platform` split at lock time. Restrict resolution to
your actual target(s) so the resolver doesn't have to find a single
solution that works on Linux *and* Windows *and* macOS at the same time:

```toml
[tool.uv]
environments = [
  "sys_platform == 'linux' and python_version >= '3.12'",
]
```

This drops the cross-platform fallback rows from your lockfile and avoids
the resolver pulling in (for example) `torch-cpu` on a CUDA-only project
just because some hypothetical macOS environment would need it.

## Skipping RAITAP's auto-bootstrap

Once your downstream `pyproject.toml` owns the extras, RAITAP's own
deps-bootstrap is redundant. See
<a href="configuration/flags.html#flag-custom-deps"><code>--custom-deps</code></a>
to disable it on the CLI, or set
<a href="configuration/flags.html#env-raitap-deps-bootstrapped"><code>_RAITAP_DEPS_BOOTSTRAPPED=1</code></a>
in the environment to short-circuit it for both the CLI and
`raitap.run(..., auto_install_deps=True)`.
