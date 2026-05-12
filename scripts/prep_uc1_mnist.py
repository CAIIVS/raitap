"""Prepare Marabou UC1 (MNIST MLP) artefacts.

Trains a tiny MLP on MNIST (replicated to 3 channels so the existing raitap
RGB image loader can consume it without a grayscale code path), exports it
to ONNX with a dynamic batch dim, and snapshots four representative sample
images plus a labels CSV.

Outputs are written into ``~/.cache/raitap/uc1_mlp_mnist/mlp_mnist.onnx``
and ``~/.cache/raitap/uc1_mnist_samples/`` so the matching Hydra configs
(``data=mnist_samples``, ``model=mlp_mnist``) can resolve them via
``${oc.env:HOME}`` interpolation.

Run::

    uv run -p 3.11 --extra torch-cpu --extra onnx-cpu python scripts/prep_uc1_mnist.py

Idempotent: re-running with the artefacts already present is a no-op
unless ``--retrain`` is passed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.utils import save_image

_CACHE_ROOT = Path.home() / ".cache" / "raitap"
_MODEL_DIR = _CACHE_ROOT / "uc1_mlp_mnist"
_SAMPLES_DIR = _CACHE_ROOT / "uc1_mnist_samples"
_ONNX_PATH = _MODEL_DIR / "mlp_mnist.onnx"
_LABELS_PATH = _CACHE_ROOT / "uc1_mnist_labels.csv"

# 3-channel 28x28 input — matches the raitap RGB image loader so no
# grayscale special case is required downstream. Replicating MNIST to RGB
# costs nothing here and keeps the pipeline path uniform.
_INPUT_CHANNELS = 3
_INPUT_SIZE = 28
_FLAT_DIM = _INPUT_CHANNELS * _INPUT_SIZE * _INPUT_SIZE
_HIDDEN_1 = 128
_HIDDEN_2 = 64
_NUM_CLASSES = 10
_BATCH_SIZE = 256
_EPOCHS = 3
_SEED = 0
_SAMPLE_LABELS = (0, 3, 7, 9)


class MnistMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(_FLAT_DIM, _HIDDEN_1),
            nn.ReLU(),
            nn.Linear(_HIDDEN_1, _HIDDEN_2),
            nn.ReLU(),
            nn.Linear(_HIDDEN_2, _NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.flatten(x))


def _to_rgb(x: torch.Tensor) -> torch.Tensor:
    """Replicate a single grayscale channel into three identical RGB channels."""
    if x.shape[1] == _INPUT_CHANNELS:
        return x
    return x.repeat(1, _INPUT_CHANNELS, 1, 1)


def _train(device: torch.device) -> MnistMLP:
    torch.manual_seed(_SEED)
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(
        root=str(_CACHE_ROOT / "torchvision_mnist"),
        train=True,
        download=True,
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(train_ds, batch_size=_BATCH_SIZE, shuffle=True)

    model = MnistMLP().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(_EPOCHS):
        running = 0.0
        for batch_idx, (images, labels) in enumerate(loader):
            images = _to_rgb(images.to(device))
            labels = labels.to(device)
            optimiser.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimiser.step()
            running += float(loss.item())
            if batch_idx % 50 == 0:
                print(f"  epoch {epoch + 1}/{_EPOCHS} batch {batch_idx:4d} loss={loss.item():.4f}")
        print(f"epoch {epoch + 1} avg loss = {running / len(loader):.4f}")

    model.eval()
    return model


def _export_onnx(model: MnistMLP, device: torch.device) -> None:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, _INPUT_CHANNELS, _INPUT_SIZE, _INPUT_SIZE, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(_ONNX_PATH),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"wrote {_ONNX_PATH} ({_ONNX_PATH.stat().st_size // 1024} KB)")


def _snapshot_samples() -> None:
    _SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    # Earlier revisions wrote labels.csv inside the samples dir, which
    # made the raitap image loader refuse to ingest the directory (mixed
    # image + tabular files). Sweep any stale copy out of the way.
    stale_labels = _SAMPLES_DIR / "labels.csv"
    if stale_labels.exists():
        stale_labels.unlink()
    transform = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(
        root=str(_CACHE_ROOT / "torchvision_mnist"),
        train=False,
        download=True,
        transform=transform,
    )

    chosen: dict[int, tuple[str, torch.Tensor]] = {}
    for image, label in test_ds:
        label_int = int(label)
        if label_int not in _SAMPLE_LABELS or label_int in chosen:
            continue
        filename = f"mnist_{label_int}.png"
        chosen[label_int] = (filename, image)
        if len(chosen) == len(_SAMPLE_LABELS):
            break

    missing = [label for label in _SAMPLE_LABELS if label not in chosen]
    if missing:
        raise RuntimeError(
            f"MNIST test set did not yield samples for labels {missing}; "
            "the dataset may be corrupted or incomplete — delete "
            f"{_CACHE_ROOT / 'torchvision_mnist'} and retry."
        )

    rows = ["image,label"]
    for label_int in _SAMPLE_LABELS:
        filename, tensor = chosen[label_int]
        save_image(_to_rgb(tensor.unsqueeze(0)).squeeze(0), _SAMPLES_DIR / filename)
        rows.append(f"{filename},{label_int}")
    _LABELS_PATH.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {len(chosen)} sample PNGs + labels.csv into {_SAMPLES_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining even if the ONNX artefact already exists.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    if _ONNX_PATH.exists() and not args.retrain:
        print(f"{_ONNX_PATH} exists; skipping training (pass --retrain to overwrite).")
    else:
        model = _train(device)
        _export_onnx(model, device)

    if _LABELS_PATH.exists() and not args.retrain:
        print(f"{_LABELS_PATH} exists; skipping sample snapshot.")
    else:
        _snapshot_samples()


if __name__ == "__main__":
    main()
