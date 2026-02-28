import hydra
import torch
from hydra.core.hydra_config import HydraConfig

from .configs.register import register_configs
from .configs.schema import AppConfig
from .data import load_data
from .models.loader import load_model
from .transparency import explain

register_configs()


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: AppConfig):
    print("=" * 60)
    print("RAITAP Transparency Assessment")
    print("=" * 60)
    print(f"\nExperiment: {config.experiment_name}")
    print(f"Model: {config.model.source}")
    print(f"Dataset: {config.data.name}")
    print(f"Method: {config.transparency.framework}.{config.transparency.algorithm}")
    print(f"Visualisers: {config.transparency.visualisers}")
    print(f"Output: {HydraConfig.get().runtime.output_dir}\n")

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

    # 2. Load data
    print("\nLoading data...")
    if not config.data.source:
        raise ValueError(
            "No data source specified. Set data.source in your config.\n"
            "Use a local path or a named sample set, e.g.: data=imagenet_samples"
        )
    data = load_data(config.data.source)
    print(
        f"✓ Loaded {data.shape[0]} samples from {config.data.source!r} (shape: {tuple(data.shape[1:])})"
    )

    # 3. Predict target classes so attributions reflect actual model decisions.
    with torch.no_grad():
        logits = model(data)
        targets = logits.argmax(dim=1)
    print(f"✓ Predicted classes: {targets.tolist()}")

    # 4. Run explanation
    print("\nRunning explanation...")
    result = explain(config, model, data, target=targets)

    attributions = result["attributions"]
    visualisations = result["visualisations"]
    run_dir = result["run_dir"]
    print(f"✓ Attributions shape: {attributions.shape}")  # type: ignore[union-attr]
    for name in visualisations:  # type: ignore[union-attr]
        print(f"✓ Visualisation saved: {run_dir}/{name}.png")
    print(f"✓ Metadata saved:      {run_dir}/metadata.json")

    print("\n" + "=" * 60)
    print("Assessment complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
