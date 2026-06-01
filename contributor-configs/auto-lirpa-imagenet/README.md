# auto-LiRPA certified-robustness demo

Certified L∞ robustness of a **tiny verifiable CNN** on the bundled
`imagenet_samples` subset via `AutoLiRPAAssessor` (IBP / CROWN bound
propagation), rendered to an HTML report (verdict summary + certified output
bounds).

This demo is a **Python script** (`assessment.py`), not a YAML config — see
"Why a script" below.

## Why a tiny custom CNN (not ResNet / VGG)

auto-LiRPA verifies a *plain* `nn.Module`: conv / ReLU / linear with
**non-overlapping** pools (`stride == kernel_size`) and **no Dropout**.
Off-the-shelf torchvision ImageNet models trip its bound-graph converter:

| Model | Blocker |
| --- | --- |
| ResNet / DenseNet / AlexNet | overlapping maxpool stem (k=3, s=2) → `ValueError: self.stride != self.kernel_size` |
| VGG / MobileNet / EfficientNet | `Dropout` in the classifier → graph-conversion crash |
| ViT / ConvNeXt | attention / LayerNorm — unsupported |

auto-LiRPA's own examples are small CIFAR/MNIST conv nets for the same reason.
This demo therefore builds a tiny `conv + ReLU + MaxPool(k=2,s=2)` net (random
weights — swap in trained ones for meaningful accuracy).

## Why a script, not a YAML config

raitap loads custom models from disk only as TorchScript or torchvision-arch
state-dicts. auto-LiRPA can consume **neither** — it can't trace a
`ScriptModule`, and a custom net isn't a torchvision arch. The one format that
hands it a real `nn.Module` is a full pickle (`torch.save(model, path)`), which
needs `allow_unsafe_pickle` **and** the class importable at load. The script
saves and reloads in the *same process*, keeping the class in `__main__` so the
reload resolves — no PYTHONPATH or prep step.

## Install auto-LiRPA

Git-only dependency (`auto-lirpa` extra, resolved from GitHub master via
`[tool.uv.sources]`). On Windows force UTF-8 so its `setup.py` builds:

```powershell
$env:PYTHONUTF8 = "1"
uv sync --extra torch-cpu --extra auto-lirpa --extra metrics --extra reporting
```

> **Intel XPU note.** The `auto-lirpa` extra is declared mutually exclusive with
> `torch-intel` / `onnx-intel` in `pyproject.toml`: the torch-2.8 window
> auto-LiRPA needs has no resolvable XPU wheel, and auto-LiRPA has no upstream
> XPU support. To try it on your XPU env anyway, force-install it on top of
> torch-2.11+xpu (it imports and runs on 2.11 in practice, despite its `<2.9`
> declared cap):
>
> ```powershell
> $env:PYTHONUTF8 = "1"
> uv pip install --no-deps `
>   "auto-LiRPA @ git+https://github.com/Verified-Intelligence/auto_LiRPA"
> uv pip install appdirs pyyaml tqdm graphviz
> ```
>
> If bound propagation hits `operator not implemented for XPU`, set
> `hardware=Hardware.cpu` in `assessment.py`.

## Run

```powershell
uv run --no-sync python contributor-configs/auto-lirpa-imagenet/assessment.py
```

The HTML report lands under `outputs/<date>/<time>/reports/report.html`.

## Knobs (edit `assessment.py`)

- `algorithm` — `ibp` (cheap, default) / `crown` / `crown-ibp` (L∞) or
  `crown-l2` (L2).
- `constructor={"epsilon": ...}` — perturbation radius. Bounds on a random net
  are loose, so most samples land `UNKNOWN` (sound + incomplete → never
  `FALSIFIED`).
- `TinyVerifiableNet` — keep it to conv / ReLU / non-overlapping pool / linear;
  adding Dropout or an overlapping MaxPool will break bound propagation.
