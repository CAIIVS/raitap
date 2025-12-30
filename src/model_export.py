from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset

# Load Model
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
torch_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

print('Model loaded.')

# Load Dataset
dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

print('Dataset loaded.')

# Preprocessing
inputs = processor(image, return_tensors="pt")
example_input = inputs['pixel_values']

print('example_inputs:', example_input)

# Export model to ONNX
onnx_program = torch.onnx.export(torch_model, example_input, dynamo=True)

onnx_program.save("models/resnet_50.onnx")

print('Model exported.')