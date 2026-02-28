"""
Example: Config-driven workflow with Hydra

This example demonstrates how to use the transparency module
with Hydra configuration files (similar to run.py).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from raitap.transparency import create_explainer, method_from_config
from raitap.transparency.visualisers import ImageHeatmapvisualiser


def main():
    # Simulate config values (normally from Hydra)
    config = SimpleNamespace(
        transparency=SimpleNamespace(
            framework="captum",
            algorithm="Saliency",
            output_dir="outputs/config_example",
        )
    )

    # Create a simple model
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10)
    )
    model.eval()

    # Create sample images
    images = torch.randn(8, 3, 32, 32)

    # Translate config → registry (Hydra integration pattern)
    method = method_from_config(config.transparency)

    # Create explainer
    explainer = create_explainer(method, modality="image")

    # Compute attributions
    attributions = explainer.explain(model, images, target=0)

    # Visualize and save
    visualiser = ImageHeatmapvisualiser()
    output_path = f"{config.transparency.output_dir}/result.png"
    visualiser.save(attributions, output_path, inputs=images)

    print(f"Using method: {method.framework}.{method.algorithm}")
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    main()
