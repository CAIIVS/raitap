# auto-LiRPA certified-robustness demo

Certified L∞ robustness via `AutoLiRPAAssessor` (CROWN bound propagation),
rendered to an HTML report (verdict summary + certified output bounds). It is a
**Python script** (`assessment.py`), not a YAML config — see below.

At the default epsilon you get a genuine **VERIFIED / UNKNOWN mix** (~half each):
the report actually shows certified-robust samples, not a wall of `UNKNOWN`.

## Why it trains a tiny net on a synthetic dataset

Verification is against the **ground-truth label**: a sample is `VERIFIED` only
when the true class's certified lower bound beats every other class's upper bound
across the whole perturbation budget. A *randomly-initialised* net mispredicts
every sample, so the true-label logit is never the maximum and **nothing can
ever verify** — everything lands `UNKNOWN`. So, like auto-LiRPA's own
CIFAR/MNIST examples, the demo trains first.

To stay self-contained it synthesises a tiny labelled dataset (4 colour classes,
3×32×32, written as PNGs + `labels.csv` under `~/.cache/raitap/auto_lirpa_demo/`)
and trains a small net on it in a few hundred steps.

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
uv run --no-sync python `
  contributor-configs/auto-lirpa-demo/assessment.py
```

The HTML report lands under `outputs/<date>/<time>/reports/report.html`.

## Knobs (edit `assessment.py`)

- `epsilon` — perturbation radius. `0.025` gives the even VERIFIED/UNKNOWN
  split; raise it for more `UNKNOWN`, lower it (e.g. `0.005`) to verify (almost)
  everything. Sound + incomplete → never `FALSIFIED`.
- `algorithm` — `crown` (tight, default) / `ibp` (cheap, much looser) /
  `crown-ibp` (L∞) or `crown-l2` (L2).
- `TinyVerifiableNet` — keep it to conv / ReLU / non-overlapping pool / linear;
  adding Dropout or an overlapping MaxPool will break bound propagation.
