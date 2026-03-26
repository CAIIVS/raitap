#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from raitap.configs.schema import (
    AppConfig,
    DataConfig,
    MetricsConfig,
    ModelConfig,
    TrackingConfig,
    TransparencyConfig,
)
from raitap.run import run

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"


def build_parser() -> argparse.ArgumentParser:
    default_image = Path.home() / ".cache" / "raitap" / "imagenet_samples" / "golden_retriever.jpg"
    parser = argparse.ArgumentParser(
        description="Run a local RAITAP + MLflow smoke test with a pretrained model."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=default_image,
        help=f"Local image to explain (default: {default_image})",
    )
    parser.add_argument(
        "--model",
        default="resnet50",
        help="Torchvision model name or local model path (default: resnet50)",
    )
    parser.add_argument(
        "--experiment-name",
        default="smoke-test",
        help="RAITAP assessment name and MLflow experiment name (default: smoke-test)",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help=(
            "MLflow tracking URI. Defaults to a local MLflow server at "
            f"{DEFAULT_TRACKING_URI}, intended for a SQLite-backed setup."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "smoke-manual",
        help="Directory for local artifacts when not running via Hydra",
    )
    parser.add_argument(
        "--log-model",
        action="store_true",
        help="Also log the loaded PyTorch model to MLflow",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mpl_cache = REPO_ROOT / ".cache" / "matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    tracking_uri = args.tracking_uri

    if not image_path.exists():
        parser.error(
            f"Image file not found: {image_path}\n"
            "Tip: run the app once with data=imagenet_samples to populate the demo cache, "
            "or pass --image /path/to/local/image.jpg"
        )

    config = AppConfig(
        experiment_name=args.experiment_name,
        model=ModelConfig(source=args.model),
        data=DataConfig(
            name=f"{image_path.stem}_smoke",
            source=str(image_path),
        ),
        transparency={
            "smoke_captum": TransparencyConfig(
                _target_="CaptumExplainer",
                algorithm="IntegratedGradients",
                visualisers=[{"_target_": "CaptumImageVisualiser"}],
            )
        },
        metrics=MetricsConfig(
            _target_="ClassificationMetrics",
            task="multiclass",
            num_classes=1000,
        ),
        tracking=TrackingConfig(
            output_forwarding_url=tracking_uri,
            log_model=args.log_model,
        ),
        fallback_output_dir=str(output_dir),
    )

    status = "FAILED"
    try:
        run(config)
        status = "FINISHED"
    finally:
        pass

    print(f"Smoke test finished with status={status}")
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"Local artifacts: {output_dir}")
    return 0 if status == "FINISHED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
