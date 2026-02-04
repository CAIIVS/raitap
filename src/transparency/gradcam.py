'''
This Script executes GradCam on an ONNX Model
'''
###################################################################################
from transformers import AutoImageProcessor, ResNetForImageClassification
from datasets import load_dataset
import torch
from torchvision import transforms
from captum.attr import GuidedGradCam
from PIL import Image

###################################################################################

def custom_image_process(image: str, device= 'cuda' if torch.cuda.is_available() else 'cpu', image_size=224):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[0.225, 0.224, 0.229]),
        ]
    )
    image = image.convert("RGB")  # Convert to RGB
    image = transform(image).unsqueeze(0).to(device)
    return image

###################################################################################

# Load PyTorch Model
pytorch_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

# Print the Model Layers to Allow Layer Selection for GradCam
print('Layer:', pytorch_model.resnet.encoder.stages[3].layers[2].layer[2])
target_layer = pytorch_model.resnet.encoder.stages[3].layers[2].layer[2].convolution

# Load Data
dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

# Load GradCam
explainer = GuidedGradCam(pytorch_model, layer=target_layer)

# Apply GradCam
## Preprocess
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
inputs = processor(image, return_tensors="pt")
example_input = inputs['pixel_values']
print('_'*80)
print('Original Processor:', example_input)

input_tensor = custom_image_process(image)
example_input = input_tensor
print('_'*80)
print('Custom Processor:', example_input)

## Get Relevant Class
with torch.no_grad():
    logits = pytorch_model(example_input).logits

max_idx = logits.argmax(-1).item()

## Get Attribution
attribution = explainer.attribute(example_input, target=max_idx)

# Display Result


##########################
'''
import numpy as np
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch
import mlflow
from ultralytics import YOLO
from torch.utils.data import ConcatDataset, DataLoader
from ultralytics.data.dataset import YOLODataset

from captum.attr import GuidedGradCam
from captum.attr import visualization

# GRADIENTS METHOD
'''
'''
# Cast from uint8 to float32
norm_img = img/255.0

# Enable GPU support
norm_img = norm_img.to(device='cuda')

# Get class with maximum predicted probability
out = model(img)  # See https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results for a description of the YOLO model output.

max_idx = torch.argmax(out[0].probs.data).item()
print('Predicted class: ', max_idx)
'''
'''
# Extract backbone model and final layer
bb_model = model.model.model
layer = bb_model[9].conv.conv

# Create gradients explainer
explainer = GuidedGradCam(bb_model, layer=layer)

# Call attribution function (size matches input size, Nx3x640x640)
attribution = explainer.attribute(norm_img, target=max_idx)

# Create and save attribution image
default_cmap = LinearSegmentedColormap.from_list('custom black', 
                                                [(0, '#ffffff'),
                                                (0.25, '#000000'),
                                                (1, '#000000')], N=256)

visualization.visualize_image_attr(np.transpose(attribution.squeeze().cpu().detach().numpy(), (1,2,0)),
                        np.transpose(img.squeeze().cpu().detach().numpy(), (1,2,0)),
                        method='heat_map',
                        cmap=default_cmap,
                        show_colorbar=True,
                        sign='positive')

fig = plt.gcf()
fig.savefig("gradcam_gradients.png")
'''