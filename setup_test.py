# setup_test.py
import json
import os
import torch
import torch.nn as nn
from torchvision import models

os.makedirs("model", exist_ok=True)

classes = [
    "Corn_(maize)___Common_rust",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight"
]

# Create dummy model
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, len(classes))
)

torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': classes,
    'class_to_idx': {n: i for i, n in enumerate(classes)},
    'num_classes': len(classes)
}, "model/agrivision_model.pth")

print("✅ Dummy model created!")
print("⚠️  Run 'python train_model_pytorch.py' to train the real model.")