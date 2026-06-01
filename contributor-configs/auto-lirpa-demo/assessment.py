"""auto-LiRPA certified-robustness demo (self-contained Python script).

auto-LiRPA verifies a *plain* ``nn.Module`` — conv / ReLU / linear with
**non-overlapping** pools and **no Dropout**. Off-the-shelf torchvision ImageNet
models don't qualify (ResNet's overlapping maxpool stem raises
``stride != kernel_size``; VGG / MobileNet / EfficientNet classifiers carry
Dropout the bound-graph converter rejects). So this demo builds a tiny
verifiable CNN in-process, saves it, and runs the raitap pipeline against the
bundled ``imagenet_samples`` subset — emitting an HTML report with the
formal-verification visualisers (verdict summary + certified output bounds).

Why a script and not a YAML config: raitap loads custom models from disk only
as TorchScript or torchvision-arch state-dicts. auto-LiRPA can consume neither
(it can't trace a ``ScriptModule``, and a custom net isn't a torchvision arch).
The one format that hands auto-LiRPA a real ``nn.Module`` is a full pickle,
which needs ``allow_unsafe_pickle=True`` *and* the class importable at load
time. Saving and loading inside this single process keeps the class in
``__main__``, so the reload resolves cleanly — no PYTHONPATH or prep step.

The weights are random (untrained) — fine for exercising the verifier and the
report. Swap in trained weights for meaningful clean accuracy.

Run (after installing auto-LiRPA — see README.md):

    uv run --no-sync python contributor-configs/auto-lirpa-imagenet/assessment.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from raitap import AppConfig, Hardware, run
from raitap.data import DataConfig, LabelsConfig
from raitap.metrics import multiclass_classification
from raitap.models import ModelConfig
from raitap.reporting import html
from raitap.robustness import auto_lirpa, output_bounds_pinned, verdict_summary

_MODEL_PATH = Path.home() / ".cache" / "raitap" / "auto_lirpa_demo" / "tiny_cnn.pt"


class TinyVerifiableNet(nn.Module):
    """Conv / ReLU + non-overlapping ``MaxPool(k=2, s=2)`` + linear — the op set
    auto-LiRPA's bound propagators support. ``3x224x224 -> num_classes``."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 -> 28
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(8 * 7 * 7, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _save_model() -> Path:
    torch.manual_seed(0)  # deterministic random init (untrained — demo only).
    model = TinyVerifiableNet().eval()
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, _MODEL_PATH)  # full pickle; reloaded in-process below.
    return _MODEL_PATH


def build_config(model_path: Path) -> AppConfig:
    return AppConfig(
        hardware=Hardware.gpu,
        experiment_name="auto-lirpa-imagenet-demo",
        model=ModelConfig(source=str(model_path)),
        data=DataConfig(
            name="imagenet_samples",
            source="imagenet_samples",
            forward_batch_size=2,
            labels=LabelsConfig(source="imagenet_samples", id_column="image", column="label"),
        ),
        metrics=multiclass_classification(num_classes=1000),
        robustness={
            # IBP is cheap; swap algorithm to "crown" for tighter certificates.
            "auto_lirpa_ibp": auto_lirpa(
                algorithm="ibp",
                constructor={"epsilon": 0.002},
                visualisers=[verdict_summary(), output_bounds_pinned(max_samples=4)],
            ),
        },
        reporting=html(filename="report"),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model_path = _save_model()
    # allow_unsafe_pickle: the checkpoint is a full pickle we just wrote ourselves
    # this process, so the unsafe-load is loading our own trusted file.
    run(build_config(model_path), allow_unsafe_pickle=True)


if __name__ == "__main__":
    main()
