"""Generate candidate detection labels for the Udacity dashcam samples.

Runs ``fasterrcnn_resnet50_fpn_v2`` (the same backbone the contributor config
ships with) on the 4 ``UdacitySelfDriving`` sample images, dumps high-
confidence predictions as a RAITAP detection-labels JSON. Hand-edit the
output to trim false positives / fix labels / add missed boxes before
checking the file in as ``artifacts/udacity-boxes.json``.

Usage::

    uv run --extra torch-intel python \\
        contributor-configs/fasterrcnn-udacity/scripts/generate_candidate_labels.py

The script writes ``artifacts/udacity-boxes.candidate.json`` next to the
real labels file. Do NOT check the candidate file in — diff it against
``udacity-boxes.json`` to spot drift after model updates.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torchvision.io import read_image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

SCORE_THRESHOLD = 0.5  # drop low-confidence candidates; tune as needed
SAMPLES_DIR = Path.home() / ".cache" / "raitap" / "UdacitySelfDriving"
SAMPLE_FILES = [
    "straight_lines1.jpg",
    "straight_lines2.jpg",
    "test1.jpg",
    "test2.jpg",
]
OUTPUT = Path(__file__).resolve().parent.parent / "labels" / "udacity-boxes.candidate.json"


def main() -> None:
    if not SAMPLES_DIR.exists():
        raise SystemExit(
            f"Samples not cached at {SAMPLES_DIR}. Run once to fetch them:\n"
            '  uv run python -c "from raitap.data.samples import _load_sample; '
            "_load_sample('UdacitySelfDriving')\""
        )

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=SCORE_THRESHOLD)
    model.eval()
    category_names = weights.meta["categories"]

    records: list[dict[str, object]] = []
    with torch.no_grad():
        for filename in SAMPLE_FILES:
            image_path = SAMPLES_DIR / filename
            if not image_path.exists():
                raise SystemExit(f"Missing sample image: {image_path}")
            # fasterrcnn expects float tensors in [0, 1] without batch dim,
            # passed as a list so each sample can have different H/W.
            image_uint8 = read_image(str(image_path))
            image_float = image_uint8.float() / 255.0
            prediction = model([image_float])[0]
            keep = prediction["scores"] >= SCORE_THRESHOLD
            boxes = prediction["boxes"][keep].tolist()
            labels = prediction["labels"][keep].tolist()
            scores = prediction["scores"][keep].tolist()
            print(f"{filename}: {len(boxes)} candidate(s) at score>={SCORE_THRESHOLD}")
            for label_index, score, (x1, y1, x2, y2) in zip(labels, scores, boxes, strict=False):
                name = category_names[label_index] if label_index < len(category_names) else "?"
                print(
                    f"  - {name} (id={label_index}, score={score:.2f}): "
                    f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
                )
            records.append(
                {
                    "sample_id": filename,
                    "boxes": [[round(v, 1) for v in box] for box in boxes],
                    "labels": labels,
                }
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(records, indent=2))
    print(f"\nWrote {OUTPUT}")
    print("Now hand-edit, then save as labels/udacity-boxes.json.")


if __name__ == "__main__":
    main()
