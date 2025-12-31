from transformers import AutoImageProcessor, ResNetForImageClassification
import numpy as np
import torch
from datasets import load_dataset
import onnxruntime as ort
import json

# Load ONNX Model
ort_session = ort.InferenceSession('models/resnet_50.onnx')

print('Model loaded.')

# Read Label Dictionary from File
id2label = json.load(open( "examples/resnet_50_id2label.json" ))

# Load Dataset
dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

print('Dataset loaded.')

# Preprocessing
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
inputs = processor(image, return_tensors="pt")
example_input = inputs['pixel_values']

# Inference from Dataset
output_logits = ort_session.run(None, {'pixel_values': np.array(example_input)})

label_id = str(np.argmax(output_logits))

print('Predicted label:', id2label[label_id])