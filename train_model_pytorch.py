import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

# ============================================================
# GPU CONFIGURATION
# ============================================================
DEVICE = torch.device("xpu")  # Intel Arc GPU
print(f"Using device: {DEVICE}")
print(f"Device name: {torch.xpu.get_device_name(0)}")

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_PATH = "PlantVillage"
VARIANT = "color"
MODEL_SAVE_PATH = "model/agrivision_model.pth"
CLASS_INDICES_PATH = "class_indices.json"

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_WORKERS = 0  # Windows compatibility

# ============================================================
# Custom Dataset Class
# ============================================================
class PlantVillageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.class_names = sorted(dataframe['label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepath']
        label = self.dataframe.iloc[idx]['label']
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label_idx = self.class_to_idx[label]
        return image, label_idx

# ============================================================
# 1. Build file list
# ============================================================
variant_path = os.path.join(DATASET_PATH, VARIANT)

if not os.path.exists(variant_path):
    raise ValueError(f"Folder not found: {variant_path}")

disease_classes = [d for d in os.listdir(variant_path)
                   if os.path.isdir(os.path.join(variant_path, d)) and not d.startswith('.')]

print(f"\n{'='*50}")
print(f"Variant: '{VARIANT}'")
print(f"Found {len(disease_classes)} disease classes:")
for i, cls in enumerate(disease_classes):
    print(f"  {i+1}. {cls}")

data = []
for disease in disease_classes:
    disease_path = os.path.join(variant_path, disease)
    img_count = 0
    for img_name in os.listdir(disease_path):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            data.append([os.path.join(disease_path, img_name), disease])
            img_count += 1
    print(f"     └─ {img_count} images")

print(f"\nTotal images: {len(data)}")

df = pd.DataFrame(data, columns=["filepath", "label"])
CLASS_NAMES = sorted(df["label"].unique())
NUM_CLASSES = len(CLASS_NAMES)

# ============================================================
# 2. Train-validation split
# ============================================================
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)

print(f"\nTraining samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# ============================================================
# 3. Data transforms
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================================
# 4. DataLoaders
# ============================================================
train_dataset = PlantVillageDataset(train_df, train_transform)
val_dataset = PlantVillageDataset(val_df, val_transform)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=NUM_WORKERS
)

# Save class mapping
class_to_idx = train_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names_list = [idx_to_class[i] for i in range(len(idx_to_class))]

with open(CLASS_INDICES_PATH, "w") as f:
    json.dump({"class_indices": class_to_idx, "class_names": class_names_list}, f, indent=2)

print(f"\n✅ Class mapping saved: {CLASS_INDICES_PATH}")

# ============================================================
# 5. Build EfficientNetB0 model
# ============================================================
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Freeze base layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, NUM_CLASSES)
)

model = model.to(DEVICE)
print(f"\nModel: EfficientNetB0 ({sum(p.numel() for p in model.parameters()):,} params)")
print(f"Model is on: {next(model.parameters()).device}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

# ============================================================
# 6. Training functions
# ============================================================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", ncols=80)
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100*correct/total:.1f}%'})

    return running_loss / len(loader), 100 * correct / total

def validate_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", ncols=80)
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100*correct/total:.1f}%'})

    return running_loss / len(loader), 100 * correct / total

# ============================================================
# 7. Training loop
# ============================================================
print(f"\n{'='*50}")
print(f"STARTING TRAINING ON INTEL ARC GPU")
print(f"{'='*50}")
print(f"Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE}")
print(f"{'='*50}\n")

best_acc = 0.0
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion)
    
    epoch_time = time.time() - epoch_start
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        os.makedirs("model", exist_ok=True)
        # Move model to CPU before saving
        model.cpu()
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_names': class_names_list,
            'class_to_idx': class_to_idx,
            'num_classes': NUM_CLASSES,
            'val_accuracy': val_acc
        }, MODEL_SAVE_PATH)
        model.to(DEVICE)  # Move back to GPU
        saved = " ✅ Saved!"
    else:
        saved = ""
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.0f}s | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%{saved}")

# ============================================================
# 8. Summary
# ============================================================
total_time = time.time() - start_time
print(f"\n{'='*50}")
print(f"TRAINING COMPLETE")
print(f"{'='*50}")
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Best validation accuracy: {best_acc:.1f}%")
print(f"Model saved to: {MODEL_SAVE_PATH}")
print(f"{'='*50}")