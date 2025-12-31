from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset

# Load ResNet 50 Model
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

print('Model loaded.')

# Load Dataset
dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

print('Dataset loaded.')

# Preprocessing
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
inputs = processor(image, return_tensors="pt")
example_input = inputs['pixel_values']

# Inference from Dataset
with torch.no_grad():
    logits = model(example_input).logits

predicted_label = logits.argmax(-1).item()

print('Predicted label:', model.config.id2label[predicted_label])