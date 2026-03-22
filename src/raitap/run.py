import hydra

from raitap.configs.factory_utils import resolve_run_dir

from .configs.register import register_configs
from .configs.schema import AppConfig
from .data import load_data
from .models import Model
from .tracking import (
    create_tracker,
    finalize_tracking,
    initialize_tracking,
    log_dataset_info,
)
from .transparency.factory import Explanation, create_visualisers

register_configs()


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: AppConfig) -> None:
    tracker = create_tracker(config.tracking)
    status = "FAILED"

    print_summary(config)

    try:
        initialize_tracking(tracker, config)

        print("Loading model...")
        model = Model(config)
        print(f"✓ Loaded model from {config.model.source!r}")
        model.log(tracker)

        print("\nLoading data...")
        data = load_data(config)
        n, *dims = data.shape
        print(f"✓ Loaded {n} samples from {config.data.source!r} (shape: {tuple[int, ...](dims)})")
        log_dataset_info(tracker, config, data)

        # 3. Run transparency assessment
        print("\nRunning explanation...")
        explanation = Explanation(config, model, data)
        explanation.log(tracker)

        visualisations = [
            explanation.visualise(visualiser) for visualiser in create_visualisers(config)
        ]
        for visualisation in visualisations:
            visualisation.log(tracker)

        status = "FINISHED"

        print("\n" + "=" * 60)
        print("Assessment complete!")
        print("=" * 60)
    finally:
        finalize_tracking(tracker, status=status)


def print_summary(config: AppConfig) -> None:
    print("=" * 60)
    print("RAITAP Transparency Assessment")
    print("=" * 60)
    print(f"\nExperiment: {config.experiment_name}")
    print(f"Model: {config.model.source}")
    print(f"Dataset: {config.data.name}")
    print(f"Framework: {config.transparency._target_}")
    print(f"Algorithm: {config.transparency.algorithm}")
    print(f"Visualisers: {config.transparency.visualisers}")
    print(f"Output: {resolve_run_dir(config)}\n")


if __name__ == "__main__":
    main()
