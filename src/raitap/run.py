import hydra
import torch

from .configs.register import register_configs
from .configs.schema import AppConfig
from .models.data_loader import load_images_from_directory
from .models.loader import load_pretrained_model
from .transparency import explain

register_configs()


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: AppConfig):
    print("=" * 60)
    print("RAITAP Transparency Assessment")
    print("=" * 60)
    print(f"\nExperiment: {config.experiment_name}")
    print(f"Model: {config.model.name}")
    print(f"Dataset: {config.data.name}")
    print(f"Method: {config.transparency.framework}.{config.transparency.algorithm}")
    print(f"Visualisers: {config.transparency.visualisers}")
    print(f"Output: {config.transparency.output_dir}\n")

    # 1. Load model
    print("Loading model...")
    if config.model.name:
        model = load_pretrained_model(config.model.name, pretrained=config.model.pretrained)
    else:
        raise ValueError("No model specified. Set model.name in config or use model=resnet50")
    model.eval()
    print(f"✓ Loaded {config.model.name}")

    # 2. Load data
    print("\nLoading data...")
    if not config.data.directory:
        raise ValueError(
            "No data directory specified. Set data.directory in your config.\n"
            "For a quick test run: uv run raitap data=imagenet_samples"
        )
    images = load_images_from_directory(config.data.directory)
    print(
        f"✓ Loaded {images.shape[0]} images from {config.data.directory} ({images.shape[2]}x{images.shape[3]})"
    )

    # 3. Predict target classes so attributions reflect actual model decisions.
    with torch.no_grad():
        logits = model(images)
        targets = logits.argmax(dim=1)
    print(f"✓ Predicted classes: {targets.tolist()}")

    # 4. Run explanation
    print("\nRunning explanation...")
    result = explain(config, model, images, target=targets)

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
