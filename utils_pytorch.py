import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Load disease info
with open("diseases.json", "r") as f:
    DISEASE_INFO = json.load(f)

# Load class names
with open("class_indices.json", "r") as f:
    mapping = json.load(f)
    CLASS_NAMES = mapping["class_names"]

DEVICE = torch.device("cpu")

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model(model_path="model/agrivision_model.pth"):
    """Load trained PyTorch model."""
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, checkpoint['num_classes'])
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict_disease(model, uploaded_file):
    """Predict disease from uploaded file."""
    image = Image.open(uploaded_file).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    return predicted_class, confidence.item()

def get_disease_info(disease_name):
    if disease_name in DISEASE_INFO:
        return DISEASE_INFO[disease_name]
    return {
        "description": "No detailed information available for this disease.",
        "treatment": "Please consult an agricultural expert for guidance."
    }
