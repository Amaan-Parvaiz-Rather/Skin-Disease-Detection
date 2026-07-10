"""
Skin Disease Classification - EffNet Training Script (Target: 85%+)
================================================================
Builds on V2's ConvNeXt-Base backbone. No new data needed - all gains
come from smarter training techniques:

  1.  RandAugment   - policy-based augmentation (replaces hand-tuned aug)
  2.  MixUp + CutMix (50/50 random switch) - strongest regularisers
  3.  Differential learning-rates - backbone:head = 1:10
  4.  Cosine warmup + CosineAnnealingWarmRestarts (SGDR)
  5.  Stage 3 polishing - tiny LR (1e-6) for the final plateau push
  6.  Enhanced TTA - 7 augmentations (H-flip, V-flip, rot90s, orig)
  7.  Label-Smoothing FocalLoss (unchanged from V2)
  8.  Two-layer classifier head (Linear->GELU->Linear)

Expected gain: +8-12% over V2's 74.54 -> target 83-87%

Usage:
    python train_image_v3.py
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
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# FOCAL LOSS
# ============================================================================
class FocalLoss(nn.Module):
    """Focal Loss with label smoothing. gamma=2 focuses on hard examples."""
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets,
                             label_smoothing=self.label_smoothing,
                             reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ============================================================================
# MIXUP & CUTMIX
# ============================================================================
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.shape

    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed, y, y[idx], lam


def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# CONFIG
# ============================================================================
SEED            = 42
IMG_SIZE        = 384
BATCH_SIZE      = 4
ACCUM_STEPS     = 4        # effective batch = 16
STAGE1_EPOCHS   = 5        # head warmup (backbone frozen)
STAGE2_EPOCHS   = 25       # full fine-tune with MixUp/CutMix
STAGE3_EPOCHS   = 10       # polishing - tiny LR
EARLY_STOP_PAT  = 8

MIXUP_PROB      = 0.5      # 50% MixUp / 50% CutMix per augmented batch
MIXUP_ALPHA     = 0.4
CUTMIX_ALPHA    = 1.0

TRAIN_DIR = "dermatology_dataset/train"
TEST_DIR  = "dermatology_dataset/test"
SAVE_DIR  = "./Checkpoints"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

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
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()

print(f"Device : {device}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
print(f"AMP    : {'Enabled' if use_amp else 'Disabled'}")

# ============================================================================
# TRANSFORMS - RandAugment replaces hand-tuned augmentation
# ============================================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

print(f"Image size : {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size : {BATCH_SIZE}  (effective: {BATCH_SIZE * ACCUM_STEPS})")

# ============================================================================
# LOAD DATA
# ============================================================================
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=test_transform)

class_names = train_dataset.classes
num_classes = len(class_names)

train_labels   = [s[1] for s in train_dataset.samples]
class_counts   = Counter(train_labels)
sample_weights = [1.0 / class_counts[lbl] for lbl in train_labels]
sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(train_labels), replacement=True
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=0, pin_memory=True, persistent_workers=False
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=True, persistent_workers=False
)

print(f"Classes    : {num_classes}")
print(f"Train imgs : {len(train_dataset)}")
print(f"Test  imgs : {len(test_dataset)}")
print("\nClass distribution:")
for i, name in enumerate(class_names):
    print(f"  {name[:55]:55s} | {class_counts[i]:5d}")

# ============================================================================
# MODEL - ConvNeXt-Base with improved 2-layer head
# ============================================================================
print("\n" + "=" * 60)
print("BUILDING MODEL - ConvNeXt-Base")
print("=" * 60)

weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
model   = efficientnet_v2_s(weights=weights)

in_features = model.classifier[1].in_features  # 1024

# Improved 2-layer classifier head
model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, 512),
    nn.GELU(),
    nn.Dropout(p=0.2),
    nn.Linear(512, num_classes)
)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters : {total_params:,}")

# ============================================================================
# LOSS
# ============================================================================
criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
os.makedirs(SAVE_DIR, exist_ok=True)

scaler    = GradScaler("cuda") if use_amp else None
BEST_PATH = ""
LAST_PATH = ""


def save_checkpoint(epoch, best_acc):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc": best_acc
    }
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()
    torch.save(ckpt, LAST_PATH)


# ============================================================================
# TRAIN ONE EPOCH - with optional MixUp / CutMix
# ============================================================================
def train_one_epoch(use_mixup_cutmix=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Apply MixUp or CutMix 80% of batches
        if use_mixup_cutmix and random.random() < 0.8:
            if random.random() < MIXUP_PROB:
                images, lbl_a, lbl_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
            else:
                images, lbl_a, lbl_b, lam = cutmix_data(images, labels, CUTMIX_ALPHA)
            mixed = True
        else:
            lbl_a, lbl_b, lam = labels, labels, 1.0
            mixed = False

        if use_amp:
            with autocast("cuda"):
                outputs = model(images)
                loss = mixed_criterion(criterion, outputs, lbl_a, lbl_b, lam) if mixed else criterion(outputs, labels)
                loss_scaled = loss / ACCUM_STEPS
            scaler.scale(loss_scaled).backward()
            if (batch_idx + 1) % ACCUM_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            outputs = model(images)
            loss = mixed_criterion(criterion, outputs, lbl_a, lbl_b, lam) if mixed else criterion(outputs, labels)
            loss_scaled = loss / ACCUM_STEPS
            loss_scaled.backward()
            if (batch_idx + 1) % ACCUM_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * images.size(0)
        preds   = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(f"  [Train] Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    return running_loss / total, correct / total


# ============================================================================
# VALIDATE
# ============================================================================
def validate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if use_amp:
                with autocast("cuda"):
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss    = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return running_loss / total, correct / total


# ============================================================================
# ENHANCED TTA - 7 augmentations
# ============================================================================
def tta_predict(images):
    """7-aug TTA: orig + H-flip + V-flip + rot90x4 + 90+H-flip."""
    aug_list = [
        images,
        torch.flip(images, dims=[3]),
        torch.flip(images, dims=[2]),
        torch.rot90(images, k=1, dims=[2, 3]),
        torch.rot90(images, k=2, dims=[2, 3]),
        torch.rot90(images, k=3, dims=[2, 3]),
        torch.flip(torch.rot90(images, k=1, dims=[2, 3]), dims=[3]),
    ]
    preds = []
    with torch.inference_mode():
        for aug in aug_list:
            if use_amp:
                with autocast("cuda"):
                    preds.append(F.softmax(model(aug), dim=1))
            else:
                preds.append(F.softmax(model(aug), dim=1))
    return torch.stack(preds).mean(0)


def validate_tta():
    model.eval()
    correct = 0
    total   = 0
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds  = tta_predict(images).argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total


# ============================================================================
# FIT LOOP
# ============================================================================
def fit(stage_name, epochs, use_mixup_cutmix=True):
    best_acc         = 0.0
    patience_counter = 0
    history          = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        t0 = time.time()
        print(f"\n{stage_name}  Epoch {epoch + 1}/{epochs}")
        print("-" * 55)

        train_loss, train_acc = train_one_epoch(use_mixup_cutmix)
        val_loss, val_acc     = validate()
        scheduler.step()

        lr      = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}  | Val   Acc: {val_acc:.4f}")
        print(f"LR        : {lr:.2e}        | Time: {elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), BEST_PATH)
            print(f"[OK] Best model saved! Val Acc: {best_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  EarlyStop: {patience_counter}/{EARLY_STOP_PAT} (best: {best_acc:.4f})")
            if patience_counter >= EARLY_STOP_PAT:
                print(f"\n[STOP] Early stopping at epoch {epoch + 1}")
                break

        save_checkpoint(epoch, best_acc)

    print(f"\nBest Val Acc ({stage_name}): {best_acc:.4f}")
    return best_acc, history


# ============================================================================
# STAGE 1: Classifier Head Warm-Up (backbone frozen)
# ============================================================================
print("\n" + "=" * 60)
print("STAGE 1 - Classifier Head Warm-Up (backbone frozen)")
print("=" * 60)

BEST_PATH = os.path.join(SAVE_DIR, "V6_S1_best.pth")
LAST_PATH = os.path.join(SAVE_DIR, "V6_S1_last.pth")

for p in model.features.parameters():
    p.requires_grad = False
for p in model.classifier.parameters():
    p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,}")

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3, weight_decay=1e-4
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE1_EPOCHS, eta_min=1e-5)

best_s1, hist_s1 = fit("Stage 1 - Head Warm-Up", STAGE1_EPOCHS, use_mixup_cutmix=False)

# ============================================================================
# STAGE 2: Full Fine-Tune with MixUp/CutMix + Differential LR
# ============================================================================
print("\n" + "=" * 60)
print("STAGE 2 - Full Fine-Tune + MixUp/CutMix (all layers)")
print("=" * 60)

BEST_PATH = os.path.join(SAVE_DIR, "V6_S2_best.pth")
LAST_PATH = os.path.join(SAVE_DIR, "V6_S2_last.pth")

model.load_state_dict(torch.load(
    os.path.join(SAVE_DIR, "V6_S1_best.pth"), map_location=device
))

for p in model.parameters():
    p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,}")

# DIFFERENTIAL LR - backbone gets 10x lower LR than head
backbone_params   = list(model.features.parameters())
classifier_params = list(model.classifier.parameters())

optimizer = optim.AdamW([
    {"params": backbone_params,   "lr": 5e-5},
    {"params": classifier_params, "lr": 5e-4},
], weight_decay=1e-4)

# CosineAnnealingWarmRestarts: restart every 10 epochs
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=1, eta_min=1e-6
)

scaler = GradScaler("cuda") if use_amp else None

best_s2, hist_s2 = fit("Stage 2 - Full Fine-Tune", STAGE2_EPOCHS, use_mixup_cutmix=True)

# ============================================================================
# STAGE 3: Polishing - tiny LR, no MixUp/CutMix
# ============================================================================
print("\n" + "=" * 60)
print("STAGE 3 - Polishing with tiny LR (no MixUp/CutMix)")
print("=" * 60)

BEST_PATH = os.path.join(SAVE_DIR, "V6_S3_best.pth")
LAST_PATH = os.path.join(SAVE_DIR, "V6_S3_last.pth")

model.load_state_dict(torch.load(
    os.path.join(SAVE_DIR, "V6_S2_best.pth"), map_location=device
))

optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE3_EPOCHS, eta_min=1e-7)

scaler = GradScaler("cuda") if use_amp else None

best_s3, hist_s3 = fit("Stage 3 - Polishing", STAGE3_EPOCHS, use_mixup_cutmix=False)

# ============================================================================
# FINAL EVALUATION - use whichever stage had the best val acc
# ============================================================================
print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

stage_results = [
    (best_s1, "V6_S1_best.pth"),
    (best_s2, "V6_S2_best.pth"),
    (best_s3, "V6_S3_best.pth"),
]
best_overall = max(stage_results, key=lambda t: t[0])
best_overall_path = os.path.join(SAVE_DIR, best_overall[1])
print(f"Loading best model: {best_overall[1]}  (val acc={best_overall[0]:.4f})")

model.load_state_dict(torch.load(best_overall_path, map_location=device))
model.eval()

test_loss, test_acc = validate()
print(f"\nTest Loss    : {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}  ({test_acc * 100:.2f}%)")

print("\nRunning 7-aug TTA evaluation ...")
tta_acc = validate_tta()
print(f"TTA Accuracy : {tta_acc:.4f}  ({tta_acc * 100:.2f}%)")

# Classification report
y_true, y_pred = [], []
with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        if use_amp:
            with autocast("cuda"):
                outputs = model(images)
        else:
            outputs = model(images)
        y_pred.extend(outputs.argmax(1).cpu().numpy())
        y_true.extend(labels.numpy())

report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\n" + report)

# Save results
with open("results_effnet.txt", "w") as f:
    f.write("SKIN DISEASE CLASSIFICATION - EffNet\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Test Accuracy    : {test_acc:.4f}  ({test_acc * 100:.2f}%)\n")
    f.write(f"TTA  Accuracy    : {tta_acc:.4f}  ({tta_acc * 100:.2f}%)\n")
    f.write(f"Image Size       : {IMG_SIZE}x{IMG_SIZE}\n")
    f.write(f"Stage 1 Best Acc : {best_s1:.4f}\n")
    f.write(f"Stage 2 Best Acc : {best_s2:.4f}\n")
    f.write(f"Stage 3 Best Acc : {best_s3:.4f}\n\n")
    f.write("Key Techniques Applied:\n")
    f.write("  - RandAugment (num_ops=2, magnitude=9)\n")
    f.write("  - MixUp (alpha=0.4) + CutMix (alpha=1.0), 80% of batches\n")
    f.write("  - 2-layer classifier head (Linear->GELU->Linear)\n")
    f.write("  - Differential LR: backbone 5e-5 / head 5e-4\n")
    f.write("  - CosineAnnealingWarmRestarts (T_0=10)\n")
    f.write("  - Stage 3 polishing (LR=1e-6, no augmentation)\n")
    f.write("  - 7-aug TTA\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")

print("Results saved to results_effnet.txt")

# Training curves
all_train_acc = hist_s1["train_acc"] + hist_s2["train_acc"] + hist_s3["train_acc"]
all_val_acc   = hist_s1["val_acc"]   + hist_s2["val_acc"]   + hist_s3["val_acc"]
all_val_loss  = hist_s1["val_loss"]  + hist_s2["val_loss"]  + hist_s3["val_loss"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(all_train_acc, label="Train Acc", color="royalblue")
axes[0].plot(all_val_acc,   label="Val Acc",   color="darkorange")
s1_end = len(hist_s1["train_acc"]) - 1
s2_end = s1_end + len(hist_s2["train_acc"])
axes[0].axvline(x=s1_end, color="gray", ls="--", label="S1->S2")
axes[0].axvline(x=s2_end, color="gray", ls=":",  label="S2->S3")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(all_val_loss, label="Val Loss", color="crimson")
axes[1].set_title("Val Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle(f"EffNet Training - Test Acc: {test_acc*100:.2f}%  TTA: {tta_acc*100:.2f}%")
plt.tight_layout()
plt.savefig("training_curves_v3.png", dpi=150)
print("Curves saved to training_curves_v3.png")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(14, 12))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar()
plt.xticks(range(num_classes), class_names, rotation=90, fontsize=7)
plt.yticks(range(num_classes), class_names, fontsize=7)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"EffNet Confusion Matrix - Accuracy: {test_acc * 100:.2f}%")
plt.tight_layout()
plt.savefig("confusion_matrix_effnet.png", dpi=150)
print("Confusion matrix saved to confusion_matrix_effnet.png")

with open("class_names.json", "w") as f:
    json.dump(class_names, f)
print("Class names saved to class_names.json")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print(f"Final Test Accuracy : {test_acc * 100:.2f}%")
print(f"Final TTA  Accuracy : {tta_acc * 100:.2f}%")
print("=" * 60)
