import hydra
import torch

from raitap.configs.factory_utils import resolve_run_dir
from raitap.configs.register import register_configs
from raitap.configs.schema import AppConfig
from raitap.data import Data
from raitap.models import Model
from raitap.tracking import BaseTracker
from raitap.transparency.factory import Explanation, create_visualisers
from raitap.transparency.results import ExplanationResult, VisualisationResult

register_configs()


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: AppConfig) -> None:
    print_summary(config)

    print("Loading model...")
    model = Model(config)
    print(f"Loaded model from {config.model.source!r}")

    print("\nLoading data...")
    data = Data(config)
    data_tensor = data.tensor
    n, *dims = data_tensor.shape
    print(f"Loaded {n} samples from {config.data.source!r} (shape: {tuple[int, ...](dims)})")

    [explanations, visualisations_list] = run_explanations(config, model, data, data_tensor)

    # Only use tracking if a valid tracker is configured (_target_ is present)
    tracking_config = config.tracking if hasattr(config, "tracking") else None
    has_tracker = tracking_config and hasattr(tracking_config, "_target_")

    if has_tracker:
        with BaseTracker.create_tracker(config) as tracker:
            tracker.log_config()
            if config.tracking.log_model:
                model.log(tracker)
            data.log(tracker)
            for explanation in explanations:
                explanation.log(tracker)
            for visualisation in visualisations_list:
                visualisation.log(tracker)
            # for metric in metrics:
            #     metric.log(tracker)
            data.log(tracker)
            for explanation in explanations:
                explanation.log(tracker)
            for visualisation in visualisations_list:
                visualisation.log(tracker)
            # for metric in metrics:
            #     metric.log(tracker)

    print("\n" + "=" * 60)
    print("Assessment complete!")
    print("=" * 60)


def run_explanations(
    config: AppConfig, model: Model, data: Data, data_tensor: torch.Tensor
) -> tuple[list[ExplanationResult], list[VisualisationResult]]:

    explainers = config.transparency.items()
    if not explainers:
        raise ValueError("No explainers configured")

    print("\nComputing explanations...")
    explanations = []
    visualisations_list = []

    import torch

    with torch.no_grad():
        logits = model.network(data_tensor)
        predicted_classes = logits.argmax(dim=1)
        target = predicted_classes.tolist()

    for name, explainer_config in explainers:
        print(f"  -> Running {name}...")
        explanation = Explanation(config, name, model, data_tensor, target=target)
        explanations.append(explanation)

        visualisations = [
            explanation.visualise(visualiser)
            for visualiser in create_visualisers(explainer_config)  # TODO explanation.visualise()
        ]
        visualisations_list.extend(visualisations)

    return explanations, visualisations_list


def print_summary(config: AppConfig) -> None:
    print("=" * 60)
    print("RAITAP Transparency Assessment")
    print("=" * 60)
    print(f"\nExperiment: {config.experiment_name}")
    print(f"Model: {config.model.source}")
    print(f"Dataset: {config.data.name}")
    print(f"Explainers: {list(config.transparency.keys())}")
    print(f"Output: {resolve_run_dir(config)}\n")


if __name__ == "__main__":
    main()
