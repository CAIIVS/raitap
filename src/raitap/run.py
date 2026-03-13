from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig

from .configs.register import register_configs
from .configs.schema import AppConfig
from .data import load_data
from .models import load_model
from .tracking import (
    create_tracker,
    finalize_tracking,
    initialize_tracking,
    log_artifact_directory,
    log_dataset_info,
)
from .tracking.helpers import log_model_artifact
from .transparency import explain

register_configs()


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: AppConfig):
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    tracker = create_tracker(config.tracking)
    status = "FAILED"

    print("=" * 60)
    print("RAITAP Transparency Assessment")
    print("=" * 60)
    print(f"\nExperiment: {config.experiment_name}")
    print(f"Model: {config.model.source}")
    print(f"Dataset: {config.data.name}")
    print(f"Framework: {config.transparency._target_}")
    print(f"Algorithm: {config.transparency.algorithm}")
    print(f"Visualisers: {config.transparency.visualisers}")
    print(f"Output: {output_dir}\n")

    try:
        initialize_tracking(tracker, config, output_dir)

        # 1. Load model
        print("Loading model...")
        if not config.model.source:
            raise ValueError(
                "No model specified. Set model.source in your config.\n"
                "  model.source: path/to/your_model.pth   (custom model)\n"
                "  model.source: resnet50                 (built-in demo model)"
            )
        model = load_model(config.model.source)
        print(f"✓ Loaded model from {config.model.source!r}")
        log_model_artifact(tracker, model)

        # 2. Load data
        print("\nLoading data...")
        if not config.data.source:
            raise ValueError(
                "No data source specified. Set data.source in your config.\n"
                "Use a local path or a named sample set, e.g.: data=imagenet_samples"
            )
        data = load_data(config.data.source)
        n, *dims = data.shape
        print(f"✓ Loaded {n} samples from {config.data.source!r} (shape: {tuple(dims)})")
        log_dataset_info(tracker, config, data)

        # 3. Run transparency assessment
        print("\nRunning explanation...")
        transparency_dir = output_dir / "transparency"
        explain(config, model, data, output_dir=transparency_dir)
        log_artifact_directory(tracker, transparency_dir, artifact_path="transparency")

        status = "FINISHED"

        print("\n" + "=" * 60)
        print("Assessment complete!")
        print("=" * 60)
    finally:
        finalize_tracking(tracker, status=status)


if __name__ == "__main__":
    main()
