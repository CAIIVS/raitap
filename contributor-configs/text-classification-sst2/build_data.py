"""Generate the fixture data for the text-classification SST-2 smoke config (#340).

Writes a tiny hand-labelled sentiment dataset the assessment tokenises and
explains:

- ``artifacts/reviews.csv`` — ``text`` column of short movie-style reviews
- ``artifacts/labels.csv``  — ``label`` column (0 = negative, 1 = positive),
  row-aligned to ``reviews.csv``

Run once before the assessment (artifacts are gitignored, so this is needed to
(re)generate them):

    uv run --extra text python contributor-configs/text-classification-sst2/build_data.py
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"

# (text, label) — 0 = negative, 1 = positive. Short, unambiguous sentiment so
# the SST-2 model's predictions are stable and the token attributions readable.
ROWS: list[tuple[str, int]] = [
    ("An absolute delight from start to finish.", 1),
    ("A dull, lifeless slog I could not wait to end.", 0),
    ("Brilliant performances and a heartfelt story.", 1),
    ("Painfully boring and utterly forgettable.", 0),
    ("A charming, funny, and genuinely moving film.", 1),
    ("The worst movie I have seen in years.", 0),
]


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)

    reviews_path = ARTIFACTS / "reviews.csv"
    with reviews_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text"])
        for text, _label in ROWS:
            writer.writerow([text])

    labels_path = ARTIFACTS / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"])
        for _text, label in ROWS:
            writer.writerow([label])

    print(f"Wrote {reviews_path} and {labels_path}")


if __name__ == "__main__":
    main()
