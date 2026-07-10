import os
import random
import time
import zipfile

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from sklearn.metrics import accuracy_score, classification_report
# Define local paths for the dataset
TRAIN_DIR = "dermatology_dataset/train"
TEST_DIR = "dermatology_dataset/test"

print("Train:", os.path.exists(TRAIN_DIR))
print("Test :", os.path.exists(TEST_DIR))
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
IMG_SIZE = 300
BATCH_SIZE = 16

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),      # Add vertical flips
    transforms.RandomRotation(30),             # Increase rotation to 30 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Add minor translation/scaling
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.2)            # Cutout augmentation
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

print("Transforms ready.")
print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

class_names = train_dataset.classes
num_classes = len(class_names)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False
)

print("Classes:", num_classes)
print("Train images:", len(train_dataset))
print("Test images :", len(test_dataset))
print("Batch size  :", BATCH_SIZE)
weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
model = efficientnet_v2_s(weights=weights)

in_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, num_classes)
)

model = model.to(device)

print(model.classifier)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal Parameters      : {total_params:,}")
print(f"Trainable Parameters  : {trainable_params:,}")
SAVE_DIR = "./Checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

BEST_PATH = os.path.join(SAVE_DIR, "V3_1_A_best.pth")
LAST_PATH = os.path.join(SAVE_DIR, "V3_1_A_last_checkpoint.pth")

# Freeze feature extractor
for param in model.features.parameters():
    param.requires_grad = False

# Classifier trainable
for param in model.classifier.parameters():
    param.requires_grad = True

# Calculate class weights for imbalanced dataset
from collections import Counter
class_counts = Counter(train_dataset.targets)
class_weights = [len(train_dataset) / (num_classes * class_counts[i]) for i in range(num_classes)]
class_weights = torch.FloatTensor(class_weights).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=5
)

scaler = GradScaler("cuda")

print("Stage 1 setup ready.")
print("Trainable Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("Best path:", BEST_PATH)
print("Last path:", LAST_PATH)
def save_checkpoint(epoch, best_acc):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_acc": best_acc
    }
    torch.save(checkpoint, LAST_PATH)


def load_checkpoint():
    if os.path.exists(LAST_PATH):
        checkpoint = torch.load(LAST_PATH, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        try:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        except RuntimeError:
            print("⚠️ Scaler state_dict is empty/incompatible, reinitializing scaler.")

        print(f"✅ Resuming from Epoch {checkpoint['epoch']+1}")

        return checkpoint["epoch"] + 1, checkpoint["best_acc"]

    return 0, 0.0


def train_one_epoch():

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):

            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  [Train] Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    return running_loss / total, correct / total


def validate():

    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.inference_mode():

        for batch_idx, (images, labels) in enumerate(test_loader):

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast("cuda"):

                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  [Val] Batch {batch_idx + 1}/{len(test_loader)} - Loss: {loss.item():.4f}")

    return running_loss / total, correct / total


def fit(stage_name, epochs):

    start_epoch, best_acc = load_checkpoint()

    for epoch in range(start_epoch, epochs):

        print(f"\n{stage_name} Epoch {epoch+1}/{epochs}")
        print("-"*40)

        train_loss, train_acc = train_one_epoch()

        val_loss, val_acc = validate()

        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.4f}")
        print(f"LR        : {lr:.6f}")

        # Save latest checkpoint every epoch
        save_checkpoint(epoch, best_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), BEST_PATH)
            print(f"✅ Best model saved! Val Acc: {best_acc:.4f}")

    print(f"\nBest Validation Accuracy: {best_acc:.4f}")

    return best_acc
best_stage1 = fit(
    stage_name="Stage 1 - Classifier Head",
    epochs=5
)
print("Best model saved:", os.path.exists(BEST_PATH))
print("Last checkpoint saved:", os.path.exists(LAST_PATH))

print("Best path:", BEST_PATH)
print("Last path:", LAST_PATH)
# Stage 2 paths
BEST_PATH = "./Checkpoints/V3_1_Stage2_best.pth"
LAST_PATH = "./Checkpoints/V3_1_Stage2_last.pth"

# Load best Stage 1 weights
model.load_state_dict(torch.load(
    "./Checkpoints/V3_1_A_best.pth",
    map_location=device
))

# Freeze everything
for param in model.features.parameters():
    param.requires_grad = False

# Unfreeze last four feature blocks
for block in model.features[-4:]:
    for param in block.parameters():
        param.requires_grad = True

# Keep classifier trainable
for param in model.classifier.parameters():
    param.requires_grad = True

print(
    "Trainable Parameters:",
    sum(p.numel() for p in model.parameters() if p.requires_grad)
)

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=10
)

scaler = GradScaler("cuda")

print("✅ Stage 2 setup ready.")
def fit(stage_name, epochs, resume=True):

    if resume:
        start_epoch, best_acc = load_checkpoint()
    else:
        start_epoch, best_acc = 0, 0.0

    for epoch in range(start_epoch, epochs):

        print(f"\n{stage_name} Epoch {epoch+1}/{epochs}")
        print("-"*40)

        train_loss, train_acc = train_one_epoch()
        val_loss, val_acc = validate()

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.4f}")
        print(f"LR        : {lr:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), BEST_PATH)
            print(f"✅ Best model saved! Val Acc: {best_acc:.4f}")

        save_checkpoint(epoch, best_acc)

    print(f"\nBest Validation Accuracy: {best_acc:.4f}")
    return best_acc
best_stage2 = fit(
    stage_name="Stage 2 - Last Blocks Fine Tune",
    epochs=10,
    resume=False
)
model.load_state_dict(torch.load(BEST_PATH, map_location=device))
model.eval()

print("Best Stage 2 model loaded.")
test_loss, test_acc = validate()

print(f"Test Loss    : {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}%")
test_loader
import torch.nn.functional as F

def tta_predict(model, images):
    model.eval()

    with torch.inference_mode():

        preds = []

        # Original
        preds.append(F.softmax(model(images), dim=1))

        # Horizontal Flip
        preds.append(
            F.softmax(
                model(torch.flip(images, dims=[3])),
                dim=1
            )
        )

        # Vertical Flip
        preds.append(
            F.softmax(
                model(torch.flip(images, dims=[2])),
                dim=1
            )
        )

        # Average predictions
        preds = torch.stack(preds).mean(0)

    return preds
correct = 0
total = 0

model.eval()

with torch.inference_mode():

    for images, labels in test_loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = tta_predict(model, images)

        preds = outputs.argmax(1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

tta_acc = correct / total

print(f"\nTTA Accuracy: {tta_acc:.4f}")
print(f"TTA Accuracy: {tta_acc*100:.2f}%")
from sklearn.metrics import classification_report

y_true = []
y_pred = []

model.eval()

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)

        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(labels.numpy())

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
))
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12,10))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar()
plt.xticks(range(num_classes), class_names, rotation=90)
plt.yticks(range(num_classes), class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
import json

with open("./class_names.json", "w") as f:
    json.dump(class_names, f)

print("Class names saved.")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
))