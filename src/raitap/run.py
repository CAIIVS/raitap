import hydra

from raitap.configs.factory_utils import resolve_run_dir
from raitap.configs.register import register_configs
from raitap.configs.schema import AppConfig
from raitap.data import load_data
from raitap.models import Model
from raitap.tracking import (
    create_tracker,
    finalize_tracking,
    initialize_tracking,
    log_dataset_info,
)
from raitap.transparency.factory import Explanation, create_visualisers

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
        print(f"Loaded model from {config.model.source!r}")
        model.log(tracker)

        print("\nLoading data...")
        data = load_data(config)
        n, *dims = data.shape
        print(f"Loaded {n} samples from {config.data.source!r} (shape: {tuple[int, ...](dims)})")
        log_dataset_info(tracker, config, data)

        # 3. Run transparency assessments
        print("\nRunning explanations...")
        explanations = []
        visualisations_list = []

        import torch

        with torch.no_grad():
            logits = model.network(data)
            predicted_classes = logits.argmax(dim=1)
            target = predicted_classes.tolist()

        for name, explainer_cfg in config.explainers.items():
            print(f"  -> Running {name}...")
            explanation = Explanation(config, name, model, data, target=target)
            explanations.append(explanation)

            visualisations = [
                explanation.visualise(visualiser)
                for visualiser in create_visualisers(explainer_cfg)
            ]
            visualisations_list.extend(visualisations)

        if config.tracking.enabled:
            print("\nLogging artifacts to tracking server...")
            for explanation in explanations:
                explanation.log(tracker)
            for visualisation in visualisations_list:
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
    print(f"Explainers: {list(config.explainers.keys())}")
    print(f"Output: {resolve_run_dir(config)}\n")


if __name__ == "__main__":
    main()
