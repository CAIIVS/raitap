"""auto-LiRPA certified-robustness demo (self-contained Python script).

Verification is against the **ground-truth label**: a sample is `VERIFIED` only
when the true class's certified lower bound beats every other class's upper
bound for all perturbations in the budget. A randomly-initialised net mispredicts
every sample, so its true-label logit is never the maximum and *nothing* can ever
verify (everything lands `UNKNOWN`). So, like auto-LiRPA's own CIFAR/MNIST
examples, this demo **trains** a tiny verifiable net first.

To stay self-contained it synthesises a tiny labelled dataset (4 colour classes,
3x32x32), trains a small conv/ReLU/MaxPool/linear net on it (the op set
auto-LiRPA's bound propagators support — no overlapping pool, no Dropout), then
runs the raitap pipeline with CROWN bound propagation and emits an HTML report.
At the chosen epsilon you get a genuine VERIFIED/UNKNOWN mix (raise epsilon for
more UNKNOWN, lower it for more VERIFIED). Sound + incomplete → never FALSIFIED.

Why a script and not a YAML config: raitap loads custom models from disk only as
TorchScript or torchvision-arch state-dicts, and auto-LiRPA can consume neither
(it can't trace a ScriptModule, and a custom net isn't a torchvision arch). The
one format that hands it a real `nn.Module` is a full pickle, which needs
`allow_unsafe_pickle=True` *and* the class importable at load. Saving and loading
inside this single process keeps the class in `__main__`, so the reload resolves
cleanly — no PYTHONPATH or prep step.

Run (after installing auto-LiRPA — see README.md):

    uv run --no-sync python contributor-configs/auto-lirpa-demo/assessment.py
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from raitap import AppConfig, Hardware, run
from raitap.data import DataConfig, LabelEncoding, LabelsConfig
from raitap.metrics import multiclass_classification
from raitap.models import ModelConfig
from raitap.reporting import html
from raitap.robustness import auto_lirpa, output_bounds_pinned, verdict_summary

_CACHE = Path.home() / ".cache" / "raitap" / "auto_lirpa_demo"
_IMAGES_DIR = _CACHE / "images"
_LABELS_CSV = _CACHE / "labels.csv"
_MODEL_PATH = _CACHE / "tiny_cnn.pt"

_NUM_CLASSES = 4
_PER_CLASS = 8
_SIZE = 32
# Distinct base colour per class — an easy, separable problem the tiny net
# overfits in a few hundred steps, giving confident (verifiable) predictions.
_BASE_COLOURS = torch.tensor(
    [[0.85, 0.1, 0.1], [0.1, 0.7, 0.1], [0.1, 0.2, 0.85], [0.85, 0.8, 0.1]]
)


class TinyVerifiableNet(nn.Module):
    """Conv / ReLU + non-overlapping ``MaxPool(k=2, s=2)`` + linear — the op set
    auto-LiRPA's bound propagators support. ``3x32x32 -> num_classes``."""

    def __init__(self, num_classes: int = _NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(8 * 8 * 8, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _synthesise_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """Write PNGs + labels.csv to the cache dir; return the (quantised) train tensors.

    Images are saved as uint8 PNGs and the training tensors are quantised to the
    same 8-bit grid, so the net trains on exactly what raitap's loader will feed.
    """
    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(0)
    images: list[torch.Tensor] = []
    labels: list[int] = []
    rows: list[tuple[str, int]] = []
    # Round-robin class order (img_000=class0, img_001=class1, …) so the batch is
    # interleaved, not class-blocked — the pinned-samples chart then shows mixed
    # classes/verdicts rather than the first class only.
    order = [cls for _ in range(_PER_CLASS) for cls in range(_NUM_CLASSES)]
    for idx, cls in enumerate(order):
        base = _BASE_COLOURS[cls].view(3, 1, 1)
        noise = 0.08 * torch.randn(3, _SIZE, _SIZE, generator=generator)
        chw = (base + noise).clamp(0.0, 1.0)
        uint8 = (chw.permute(1, 2, 0) * 255).round().to(torch.uint8).numpy()
        name = f"img_{idx:03d}.png"
        Image.fromarray(uint8).save(_IMAGES_DIR / name)
        images.append(torch.from_numpy(uint8).permute(2, 0, 1).float() / 255.0)
        labels.append(cls)
        rows.append((name, cls))
    with _LABELS_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label"])
        writer.writerows(rows)
    return torch.stack(images), torch.tensor(labels)


def _prepare_model() -> None:
    inputs, targets = _synthesise_dataset()
    torch.manual_seed(0)
    model = TinyVerifiableNet()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(150):  # tiny + separable → overfits to confident predictions.
        optimiser.zero_grad()
        loss = nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        optimiser.step()
    model.eval()
    accuracy = (model(inputs).argmax(1) == targets).float().mean().item()
    logging.info("trained tiny net — train accuracy %.2f", accuracy)
    torch.save(model, _MODEL_PATH)  # full pickle; reloaded in-process below.


def build_config() -> AppConfig:
    return AppConfig(
        hardware=Hardware.gpu,
        experiment_name="auto-lirpa-demo",
        model=ModelConfig(source=str(_MODEL_PATH)),
        data=DataConfig(
            name="auto_lirpa_demo",
            source=str(_IMAGES_DIR),
            labels=LabelsConfig(
                source=str(_LABELS_CSV),
                id_column="image",
                column="label",
                encoding=LabelEncoding.index,
            ),
        ),
        metrics=multiclass_classification(num_classes=_NUM_CLASSES),
        robustness={
            # CROWN gives tight enough bounds to certify the confident samples.
            # epsilon=0.025 lands a roughly even VERIFIED/UNKNOWN split — raise it
            # for more UNKNOWN, lower it (e.g. 0.005) to verify (almost) everything.
            "auto_lirpa_crown": auto_lirpa(
                algorithm="crown",
                constructor={"epsilon": 0.025},
                visualisers=[
                    # verdict_summary: VERIFIED/UNKNOWN counts over all samples.
                    # pinned: per-sample certified intervals — the interleaved batch
                    # makes the first few a VERIFIED/UNKNOWN mix (target bar fully
                    # right of the others = VERIFIED; any overlap = UNKNOWN).
                    verdict_summary(),
                    output_bounds_pinned(max_samples=4),
                ],
            ),
        },
        reporting=html(filename="report"),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _prepare_model()
    # allow_unsafe_pickle: the checkpoint is a full pickle we just wrote ourselves
    # this process, so the unsafe-load is loading our own trusted file.
    run(build_config(), allow_unsafe_pickle=True)


if __name__ == "__main__":
    main()
