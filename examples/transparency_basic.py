"""
Example: Using RAITAP Transparency Module with Captum IntegratedGradients

This example demonstrates the basic workflow:
1. Create an explainer using the registry API
2. Compute attributions
3. Visualize and save results
"""

from __future__ import annotations

import torch
import torch.nn as nn

from raitap.transparency import create_explainer
from raitap.transparency.methods import Captum
from raitap.transparency.visualisers import ImageHeatmapvisualiser


def main():
    # Create a simple model for demonstration
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10)
    )
    model.eval()

    # Create sample images (batch of 4)
    images = torch.randn(4, 3, 32, 32)

    # Create explainer using registry API (type-safe, IDE autocomplete)
    explainer = create_explainer(Captum.IntegratedGradients, modality="image")

    # Compute attributions for target class 0
    attributions = explainer.explain(model, images, target=0)

    print(f"Attributions shape: {attributions.shape}")
    print(f"Attributions type: {type(attributions)}")

    # Visualize and save
    visualiser = ImageHeatmapvisualiser()
    visualiser.save(attributions, "outputs/example_attributions.png", inputs=images)

    print("\nVisualization saved to outputs/example_attributions.png")


if __name__ == "__main__":
    main()
