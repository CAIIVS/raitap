"""
Example: Using RAITAP Transparency with Pretrained ResNet on Real Images

This example uses a pretrained ResNet50 model with real images
so you can verify the attributions make visual sense.

Run with: uv run python examples/transparency_real_images.py [path/to/image.jpg]
Or it will use a sample image from torchvision.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

from raitap.transparency import create_explainer
from raitap.transparency.methods import GradCAM  # Convenience alias for Captum.LayerGradCam
from raitap.transparency.visualisers import ImageHeatmapvisualiser

# Top 50 ImageNet class names for reference
IMAGENET_CLASSES = {
    0: "tench",
    1: "goldfish",
    207: "golden retriever",
    208: "Labrador retriever",
    209: "Great Dane",
    235: "German shepherd",
    281: "tabby cat",
    282: "tiger cat",
    283: "Persian cat",
    284: "Siamese cat",
    285: "Egyptian cat",
    291: "lion",
    292: "tiger",
    293: "jaguar",
    294: "cheetah",
    330: "Yorkshire terrier",
    340: "zebra",
    386: "African elephant",
    387: "Indian elephant",
    404: "airliner",
    609: "convertible",
    717: "pickup truck",
    779: "school bus",
    817: "sports car",
}


def create_sample_image():
    """Create a simple test image with geometric shapes."""
    from PIL import ImageDraw

    # Create 224x224 image
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    draw = ImageDraw.Draw(img)

    # Draw a simple face-like pattern
    # Head circle
    draw.ellipse([50, 50, 174, 174], fill=(200, 180, 140), outline=(0, 0, 0), width=3)
    # Eyes
    draw.ellipse([80, 80, 95, 95], fill=(0, 0, 0))
    draw.ellipse([129, 80, 144, 95], fill=(0, 0, 0))
    # Nose
    draw.ellipse([107, 110, 117, 120], fill=(150, 100, 100))
    # Mouth
    draw.arc([75, 100, 149, 150], 0, 180, fill=(255, 0, 0), width=3)

    return img


def main():
    # Use pretrained ResNet50
    print("Loading pretrained ResNet50...")
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.eval()

    # Get preprocessing transforms
    preprocess = weights.transforms()

    # Load images
    images_pil = []
    image_names = []

    if len(sys.argv) > 1:
        # Load images from command line arguments
        for img_path in sys.argv[1:]:
            path = Path(img_path)
            if path.exists():
                print(f"Loading {path.name}...")
                img = Image.open(path).convert("RGB")
                images_pil.append(img)
                image_names.append(path.stem)
            else:
                print(f"Warning: {img_path} not found, skipping.")
    else:
        # Create a sample image
        print("No images provided. Creating a sample image...")
        print(
            "(To use your own images: uv run python examples/transparency_real_images.py path/to/image.jpg)"  # noqa: E501
        )
        img = create_sample_image()
        images_pil.append(img)
        image_names.append("sample_face")

    if not images_pil:
        print("No images to process. Exiting.")
        return

    # Preprocess images
    images_tensor = []
    for img in images_pil:
        img_tensor = preprocess(img).unsqueeze(0)
        images_tensor.append(img_tensor)

    # Stack into batch
    batch_tensor = torch.cat(images_tensor, dim=0)
    print(f"\nBatch shape: {batch_tensor.shape}")

    # Get predictions
    with torch.no_grad():
        outputs = model(batch_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_classes = torch.topk(probs, k=5, dim=1)

    print("\n" + "=" * 60)
    print("MODEL PREDICTIONS:")
    print("=" * 60)
    for _i, (name, top_prob, top_class) in enumerate(
        zip(image_names, top_probs, top_classes, strict=False)
    ):
        print(f"\n{name.upper()}:")
        for j in range(5):
            class_idx = int(top_class[j].item())
            prob = top_prob[j].item()
            class_name = IMAGENET_CLASSES.get(class_idx, f"class_{class_idx}")
            print(f"  {j + 1}. {class_name:30s} {prob:6.1%}")

    # Create explainer
    print("\n" + "=" * 60)
    print("COMPUTING ATTRIBUTIONS...")
    print("=" * 60)

    # GradCAM - better for visualizing CNNs
    # For ResNet, use the last convolutional layer (layer4)
    # NOTE: layer must be passed when CREATING the explainer, not in explain()
    explainer = create_explainer(
        GradCAM,  # Alias for Captum.LayerGradCam
        modality="image",
        layer=model.layer4,  # Constructor arg for GradCAM
    )

    # Get top predicted class for each image
    predicted_classes = top_classes[:, 0]

    # Compute attributions
    attributions = explainer.explain(model, batch_tensor, target=predicted_classes)

    print(f"Attributions shape: {attributions.shape}")

    # Visualize
    visualiser = ImageHeatmapvisualiser()

    # Save visualization
    output_path = "outputs/real_images_attributions.png"
    visualiser.save(attributions, output_path, inputs=batch_tensor)

    print(f"\n✅ Visualization saved to {output_path}")
    print("\nThe heatmap shows which parts of the image were important for the prediction.")
    print("Red/yellow areas = high importance, Dark areas = low importance")
    print("\nOpen the image to verify the attributions make sense!")
    print(f"\nTo view: explorer {output_path}")


if __name__ == "__main__":
    main()
