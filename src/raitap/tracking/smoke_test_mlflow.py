#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raitap.configs.schema import (  # noqa: E402
    AppConfig,
    DataConfig,
    MetricsConfig,
    ModelConfig,
    TrackingConfig,
    TransparencyConfig,
)
from raitap.data import load_data  # noqa: E402
from raitap.metrics import evaluate_and_log as evaluate_metrics  # noqa: E402
from raitap.models import load_model  # noqa: E402
from raitap.tracking import (  # noqa: E402
    create_tracker,
    finalize_tracking,
    initialize_tracking,
    log_artifact_directory,
    log_dataset_info,
)
from raitap.tracking.helpers import log_model_artifact  # noqa: E402
from raitap.transparency import explain  # noqa: E402

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
        transparency=TransparencyConfig(
            _target_="CaptumExplainer",
            algorithm="IntegratedGradients",
            visualisers=[{"_target_": "CaptumImageVisualiser"}],
        ),
        metrics=MetricsConfig(
            _target_="ClassificationMetrics",
            task="multiclass",
            num_classes=1000,
        ),
        tracking=TrackingConfig(
            enabled=True,
            tracking_uri=tracking_uri,
            log_model=args.log_model,
        ),
        fallback_output_dir=str(output_dir),
    )

    tracker = create_tracker(config.tracking)
    status = "FAILED"

    try:
        initialize_tracking(tracker, config, output_dir)

        if not config.model.source:
            raise ValueError("No model source specified.")
        model_source: str = config.model.source
        model = load_model(model_source)
        log_model_artifact(tracker, model)

        if not config.data.source:
            raise ValueError("No data source specified.")
        data_source: str = config.data.source
        data = load_data(data_source)
        log_dataset_info(tracker, config, data)

        with torch.no_grad():
            logits = model(data)
            if logits.ndim < 2:
                raise ValueError(
                    "Smoke test expects classifier logits of shape (N, C); "
                    f"got {tuple(logits.shape)}"
                )
            num_classes = int(logits.shape[1])
            config.metrics.num_classes = num_classes
            predicted_classes = logits.argmax(dim=1)
            target = predicted_classes.tolist()

        metrics_dir = output_dir / "metrics"
        evaluate_metrics(
            config,
            predicted_classes,
            predicted_classes,
            logger=tracker,
            output_dir=metrics_dir,
        )

        transparency_dir = output_dir / "transparency"
        explain(
            config,
            model,
            data,
            output_dir=transparency_dir,
            target=target,
        )
        log_artifact_directory(tracker, transparency_dir, artifact_path="transparency")

        status = "FINISHED"
    finally:
        finalize_tracking(tracker, status=status)

    print(f"Smoke test finished with status={status}")
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"Local artifacts: {output_dir}")
    return 0 if status == "FINISHED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
