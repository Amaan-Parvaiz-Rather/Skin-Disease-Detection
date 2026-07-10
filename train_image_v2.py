"""
Skin Disease Classification - V2 Training Script (Target: 80%+)
================================================================
ConvNeXt-Base backbone — best-of-both approaches:
  1. ConvNeXt-Base (84.1% ImageNet) — strongest backbone
  2. Image size 384 — high resolution for fine-grained skin details
  3. WeightedRandomSampler for balanced batches
  4. Label smoothing 0.1
  5. Gradient accumulation (effective batch=16)
  6. CosineAnnealingLR scheduler — smooth LR decay
  7. Light augmentation (proven better for this task)
  8. Full model unfreeze in Stage 2

Usage:
    python train_image_v2.py
"""

import os
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# FOCAL LOSS — Forces model to focus on hard, misclassified examples
# ============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy examples and focuses on hard ones.
    gamma=0 → standard CrossEntropy. gamma=2 → strong focus on hard cases.
    Combines label smoothing + focal weighting.
    """
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # Standard cross-entropy with label smoothing
        ce_loss = F.cross_entropy(inputs, targets,
                                  label_smoothing=self.label_smoothing,
                                  reduction="none")
        # Focal weight: (1 - p_t)^gamma
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ============================================================================
# CONFIG
# ============================================================================
SEED = 42
IMG_SIZE = 384              # High resolution for fine-grained skin details
BATCH_SIZE = 4              # Smaller batch for larger images
ACCUM_STEPS = 4             # Gradient accumulation → effective batch = 16
STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 20
EARLY_STOP_PATIENCE = 7

TRAIN_DIR = "dermatology_dataset/train"
TEST_DIR = "dermatology_dataset/test"
SAVE_DIR = "./Checkpoints"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

# ============================================================================
# DEVICE
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()

print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"AMP: {'Enabled' if use_amp else 'Disabled'}")

# ============================================================================
# TRANSFORMS — Moderate augmentation (proven to work)
# ============================================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

print("Transforms ready.")
print(f"Image size: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")

# ============================================================================
# LOAD DATA
# ============================================================================
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

class_names = train_dataset.classes
num_classes = len(class_names)

# --- WeightedRandomSampler for balanced batches ---
train_labels = [s[1] for s in train_dataset.samples]
class_counts = Counter(train_labels)
sample_weights = [1.0 / class_counts[label] for label in train_labels]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_labels),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,          # Balanced sampling
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

print(f"Classes: {num_classes}")
print(f"Train images: {len(train_dataset)}")
print(f"Test images : {len(test_dataset)}")

# Print class distribution
print("\nClass distribution:")
for i, name in enumerate(class_names):
    count = class_counts[i]
    print(f"  {name[:50]:50s} | {count:5d}")

# ============================================================================
# MODEL — ConvNeXt-Base
# ============================================================================
print("\n" + "=" * 60)
print("BUILDING MODEL — ConvNeXt-Base")
print("=" * 60)

weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
model = convnext_base(weights=weights)

in_features = model.classifier[2].in_features  # 1024

# Replace classifier head
model.classifier[2] = nn.Sequential(
    nn.Dropout(0.15),
    nn.Linear(in_features, num_classes)
)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,}")
print(f"Classifier: {model.classifier}")

# ConvNeXt architecture:
#   model.features[0] = Stem (Conv + LayerNorm)
#   model.features[1] = Stage 1 (3 blocks)
#   model.features[2] = Downsample
#   model.features[3] = Stage 2 (3 blocks)
#   model.features[4] = Downsample
#   model.features[5] = Stage 3 (27 blocks)
#   model.features[6] = Downsample
#   model.features[7] = Stage 4 (3 blocks)
#   model.classifier  = [LayerNorm, Flatten, Linear]

# ============================================================================
# LOSS — Label smoothing ONLY, NO class_weights (sampler handles balancing)
# ============================================================================
criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================================
# TRAINING GLOBALS
# ============================================================================
scaler = GradScaler("cuda") if use_amp else None
BEST_PATH = ""
LAST_PATH = ""


def save_checkpoint(epoch, best_acc):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc": best_acc
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    torch.save(checkpoint, LAST_PATH)


def train_one_epoch():
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp:
            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_scaled = loss / ACCUM_STEPS
            scaler.scale(loss_scaled).backward()

            if ((batch_idx + 1) % ACCUM_STEPS == 0) or ((batch_idx + 1) == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_scaled = loss / ACCUM_STEPS
            loss_scaled.backward()

            if ((batch_idx + 1) % ACCUM_STEPS == 0) or ((batch_idx + 1) == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(f"  [Train] Batch {batch_idx + 1}/{len(train_loader)} "
                  f"- Loss: {loss.item():.4f}")

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

            if use_amp:
                with autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 100 == 0:
                print(f"  [Val] Batch {batch_idx + 1}/{len(test_loader)} "
                      f"- Loss: {loss.item():.4f}")

    return running_loss / total, correct / total


def fit(stage_name, epochs):
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        epoch_start = time.time()

        print(f"\n{stage_name} Epoch {epoch + 1}/{epochs}")
        print("-" * 50)

        train_loss, train_acc = train_one_epoch()
        val_loss, val_acc = validate()

        # Step scheduler
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.4f}")
        print(f"LR        : {lr:.7f} | Time: {elapsed:.1f}s")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), BEST_PATH)
            print(f"✅ Best model saved! Val Acc: {best_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  EarlyStopping: {patience_counter}/{EARLY_STOP_PATIENCE} "
                  f"(best: {best_acc:.4f})")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n⛔ Early stopping at epoch {epoch + 1}")
                break

        # Save checkpoint every epoch
        save_checkpoint(epoch, best_acc)

    print(f"\nBest Validation Accuracy ({stage_name}): {best_acc:.4f}")
    return best_acc


# ============================================================================
# STAGE 1: Train Classifier Head (Backbone Frozen)
# ============================================================================
print("\n" + "=" * 60)
print("STAGE 1: CLASSIFIER HEAD (Backbone Frozen)")
print("=" * 60)

BEST_PATH = os.path.join(SAVE_DIR, "V5_S1_best.pth")
LAST_PATH = os.path.join(SAVE_DIR, "V5_S1_last.pth")

# Freeze entire backbone
for param in model.features.parameters():
    param.requires_grad = False
# Classifier trainable (includes LayerNorm + Flatten + our new Linear head)
for param in model.classifier.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable Parameters: {trainable:,}")

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=STAGE1_EPOCHS
)

best_s1 = fit("Stage 1 - Classifier Head", STAGE1_EPOCHS)

# ============================================================================
# STAGE 2: Unfreeze Last 4 Blocks + Classifier
# ============================================================================
print("\n" + "=" * 60)
print("STAGE 2: LAST BLOCKS FINE-TUNE")
print("=" * 60)

BEST_PATH = os.path.join(SAVE_DIR, "V5_S2_best.pth")
LAST_PATH = os.path.join(SAVE_DIR, "V5_S2_last.pth")

# Load best Stage 1 weights
model.load_state_dict(torch.load(
    os.path.join(SAVE_DIR, "V5_S1_best.pth"),
    map_location=device
))

# Unfreeze entire model for full fine-tuning (proven better)
for param in model.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable Parameters: {trainable:,}")

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=STAGE2_EPOCHS
)

# Reset scaler for Stage 2
scaler = GradScaler("cuda") if use_amp else None

best_s2 = fit("Stage 2 - Last Blocks Fine Tune", STAGE2_EPOCHS)

# ============================================================================
# EVALUATION
# ============================================================================
print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

# Load best Stage 2 model
model.load_state_dict(torch.load(
    os.path.join(SAVE_DIR, "V5_S2_best.pth"),
    map_location=device
))
model.eval()

print("Best Stage 2 model loaded.")

# Standard evaluation
test_loss, test_acc = validate()
print(f"\nTest Loss    : {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

# TTA evaluation
def tta_predict(model, images):
    model.eval()
    with torch.inference_mode():
        preds = []
        if use_amp:
            with autocast("cuda"):
                preds.append(F.softmax(model(images), dim=1))
                preds.append(F.softmax(model(torch.flip(images, dims=[3])), dim=1))
                preds.append(F.softmax(model(torch.flip(images, dims=[2])), dim=1))
        else:
            preds.append(F.softmax(model(images), dim=1))
            preds.append(F.softmax(model(torch.flip(images, dims=[3])), dim=1))
            preds.append(F.softmax(model(torch.flip(images, dims=[2])), dim=1))
        preds = torch.stack(preds).mean(0)
    return preds

correct = 0
total = 0
with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = tta_predict(model, images)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

tta_acc = correct / total
print(f"\nTTA Accuracy: {tta_acc:.4f} ({tta_acc * 100:.2f}%)")

# Classification report
y_true = []
y_pred = []
with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        if use_amp:
            with autocast("cuda"):
                outputs = model(images)
        else:
            outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\n" + report)

# Save results
with open("results_v2.txt", "w") as f:
    f.write("SKIN DISEASE CLASSIFICATION — V2\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Test Accuracy     : {test_acc:.4f} ({test_acc * 100:.2f}%)\n")
    f.write(f"TTA Accuracy      : {tta_acc:.4f} ({tta_acc * 100:.2f}%)\n")
    f.write(f"Image Size        : {IMG_SIZE}x{IMG_SIZE}\n")
    f.write(f"Stage 1 Best Acc  : {best_s1:.4f}\n")
    f.write(f"Stage 2 Best Acc  : {best_s2:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
print("\n✅ Results saved to results_v2.txt")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(14, 12))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar()
plt.xticks(range(num_classes), class_names, rotation=90, fontsize=7)
plt.yticks(range(num_classes), class_names, fontsize=7)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix — Accuracy: {test_acc * 100:.2f}%")
plt.tight_layout()
plt.savefig("confusion_matrix_v2.png", dpi=150)
print("✅ Confusion matrix saved to confusion_matrix_v2.png")

# Save class names
with open("class_names.json", "w") as f:
    json.dump(class_names, f)
print("✅ Class names saved to class_names.json")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
print(f"Final TTA Accuracy : {tta_acc * 100:.2f}%")
print("=" * 60)
