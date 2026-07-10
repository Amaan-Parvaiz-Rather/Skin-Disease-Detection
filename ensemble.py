import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG
# ============================================================================
IMG_SIZE = 384
BATCH_SIZE = 4
TEST_DIR = "dermatology_dataset/test"
V3_MODEL_PATH = "Checkpoints/V3_S3_best.pth"
EFFNET_MODEL_PATH = "Checkpoints/EffNet_S3_best.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# LOAD DATA
# ============================================================================
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=True
)

class_names = test_dataset.classes
num_classes = len(class_names)

print(f"Loaded test dataset: {len(test_dataset)} images across {num_classes} classes.")

# ============================================================================
# BUILD & LOAD MODEL 1: ConvNeXt-Base
# ============================================================================
print("Loading ConvNeXt-Base (V3) model...")
model1 = convnext_base(weights=None)
in_features1 = model1.classifier[2].in_features
model1.classifier[2] = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features1, 512),
    nn.GELU(),
    nn.Dropout(p=0.2),
    nn.Linear(512, num_classes)
)
if os.path.exists(V3_MODEL_PATH):
    ckpt1 = torch.load(V3_MODEL_PATH, map_location=device, weights_only=False)
    # Handle if state dict is nested under "model_state_dict"
    if "model_state_dict" in ckpt1:
        model1.load_state_dict(ckpt1["model_state_dict"])
    else:
        model1.load_state_dict(ckpt1)
    print("✅ ConvNeXt-Base loaded.")
else:
    print(f"❌ Warning: Could not find {V3_MODEL_PATH}")

model1 = model1.to(device)
model1.eval()

# ============================================================================
# BUILD & LOAD MODEL 2: EfficientNetV2-S
# ============================================================================
print("Loading EfficientNetV2-S model...")
model2 = efficientnet_v2_s(weights=None)
in_features2 = model2.classifier[1].in_features
model2.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features2, 512),
    nn.GELU(),
    nn.Dropout(p=0.2),
    nn.Linear(512, num_classes)
)
if os.path.exists(EFFNET_MODEL_PATH):
    ckpt2 = torch.load(EFFNET_MODEL_PATH, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt2:
        model2.load_state_dict(ckpt2["model_state_dict"])
    else:
        model2.load_state_dict(ckpt2)
    print("✅ EfficientNetV2-S loaded.")
else:
    print(f"❌ Warning: Could not find {EFFNET_MODEL_PATH}")

model2 = model2.to(device)
model2.eval()

# ============================================================================
# ENSEMBLE EVALUATION
# ============================================================================
print("\n" + "=" * 60)
print("STARTING ENSEMBLE EVALUATION")
print("=" * 60)

y_true = []
y_pred = []
correct = 0
total = 0

with torch.inference_mode():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Get raw logits
        if torch.cuda.is_available():
            with torch.amp.autocast('cuda'):
                out1 = model1(images)
                out2 = model2(images)
        else:
            out1 = model1(images)
            out2 = model2(images)
            
        # Convert to probabilities
        prob1 = F.softmax(out1, dim=1)
        prob2 = F.softmax(out2, dim=1)
        
        # Average probabilities
        avg_prob = (prob1 + prob2) / 2.0
        
        # Get final predictions
        preds = avg_prob.argmax(1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches...")

accuracy = correct / total
print(f"\nFinal Ensemble Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# Classification report
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\n" + report)

# Save results
with open("results_ensemble.txt", "w") as f:
    f.write("ENSEMBLE CLASSIFICATION RESULTS (ConvNeXt-Base + EfficientNetV2-S)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Ensemble Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
print("✅ Results saved to results_ensemble.txt")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(14, 12))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar()
plt.xticks(range(num_classes), class_names, rotation=90, fontsize=7)
plt.yticks(range(num_classes), class_names, fontsize=7)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Ensemble Confusion Matrix — Accuracy: {accuracy * 100:.2f}%")
plt.tight_layout()
plt.savefig("confusion_matrix_ensemble.png", dpi=150)
print("✅ Confusion matrix saved to confusion_matrix_ensemble.png")
