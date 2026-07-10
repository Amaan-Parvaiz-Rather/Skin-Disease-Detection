import os
import cv2
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_base
from torchvision.models import efficientnet_v2_s
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIG
# ============================================================================
MODE = os.getenv("MODEL_MODE", "single") # "single" or "ensemble"
V3_MODEL_PATH = os.getenv("CONVNEXT_MODEL", "Checkpoints/V5_S2_best.pth")
EFFNET_MODEL_PATH = os.getenv("EFFNET_MODEL", "Checkpoints/V6_S3_best.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 384

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Using same classes from training
try:
    with open('class_names.json', 'r') as f:
        CLASSES = json.load(f)
except Exception:
    CLASSES = [
        'Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
        'Atopic Dermatitis Photos', 'Bullous Disease Photos', 'Cellulitis Impetigo and other Bacterial Infections',
        'Eczema Photos', 'Exanthems and Drug Eruptions', 'Hair Loss Photos Alopecia and other Hair Diseases',
        'Herpes HPV and other STDs Photos', 'Light Diseases and Disorders of Pigmentation',
        'Lupus and other Connective Tissue diseases', 'Melanoma Skin Cancer Nevi and Moles',
        'Nail Fungus and other Nail Disease', 'Poison Ivy Photos and other Contact Dermatitis',
        'Psoriasis pictures Lichen Planus and related diseases', 'Scabies Lyme Disease and other Infestations and Bites',
        'Seborrheic Keratoses and other Benign Tumors', 'Systemic Disease',
        'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives',
        'Vascular Tumors', 'Vasculitis Photos', 'Warts Molluscum and other Viral Infections'
    ]

# Severity / Base Risk mapping (1-10)
SEVERITY_MAPPING = {
    'Melanoma Skin Cancer Nevi and Moles': 10,
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions': 9,
    'Lupus and other Connective Tissue diseases': 8,
    'Systemic Disease': 8,
    'Vasculitis Photos': 8,
    'Herpes HPV and other STDs Photos': 7,
    'Bullous Disease Photos': 7,
    'Scabies Lyme Disease and other Infestations and Bites': 6,
    'Cellulitis Impetigo and other Bacterial Infections': 6,
    'Vascular Tumors': 6,
    'Exanthems and Drug Eruptions': 5,
    'Psoriasis pictures Lichen Planus and related diseases': 5,
    'Atopic Dermatitis Photos': 4,
    'Light Diseases and Disorders of Pigmentation': 4,
    'Eczema Photos': 4,
    'Tinea Ringworm Candidiasis and other Fungal Infections': 4,
    'Urticaria Hives': 3,
    'Poison Ivy Photos and other Contact Dermatitis': 3,
    'Seborrheic Keratoses and other Benign Tumors': 3,
    'Warts Molluscum and other Viral Infections': 3,
    'Nail Fungus and other Nail Disease': 3,
    'Acne and Rosacea Photos': 2,
    'Hair Loss Photos Alopecia and other Hair Diseases': 2
}

NUM_CLASSES = len(CLASSES)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# ============================================================================
# LOAD MODELS
# ============================================================================
model_convnext = None
model_effnet = None

print("Loading models into memory...")

# 1. Load ConvNeXt-Base (Always loaded as primary for prediction & Grad-CAM)
model_convnext = convnext_base(weights=None)
in_features = model_convnext.classifier[2].in_features
model_convnext.classifier[2] = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, NUM_CLASSES)
)
if os.path.exists(V3_MODEL_PATH):
    ckpt = torch.load(V3_MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model_convnext.load_state_dict(state_dict)
    print("ConvNeXt loaded.")
else:
    print(f"Warning: ConvNeXt checkpoint not found at {V3_MODEL_PATH}. Using untrained weights.")

model_convnext.to(DEVICE)
model_convnext.eval()

# 2. Load EfficientNetV2-S (Only if ensemble mode)
if MODE == "ensemble":
    model_effnet = efficientnet_v2_s(weights=None)
    in_features2 = model_effnet.classifier[1].in_features
    model_effnet.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features2, 512),
        nn.GELU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, NUM_CLASSES)
    )
    if os.path.exists(EFFNET_MODEL_PATH):
        ckpt2 = torch.load(EFFNET_MODEL_PATH, map_location=DEVICE, weights_only=False)
        state_dict2 = ckpt2['model_state_dict'] if 'model_state_dict' in ckpt2 else ckpt2
        model_effnet.load_state_dict(state_dict2)
        print("EfficientNetV2-S loaded for ensemble.")
    else:
        print(f"Warning: Ensemble mode active but {EFFNET_MODEL_PATH} not found. Falling back to single model.")
        MODE = "single"
        
    model_effnet.to(DEVICE)
    model_effnet.eval()

# ============================================================================
# GRAD-CAM IMPLEMENTATION
# ============================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        score = output[:, class_idx]
        score.backward(retain_graph=True)
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pooling on the gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Multiply activations by weights
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0) # ReLU
        
        if np.max(cam) != 0:
            cam = cam / np.max(cam) # Normalize to [0,1]
            
        return cam

# Initialize Grad-CAM on ConvNeXt's last spatial feature map
# For ConvNeXt, the best spatial map is the depthwise conv in the last block
try:
    target_layer = model_convnext.features[-1][-1].block[0]
except Exception:
    # fallback to just the last feature block entirely
    target_layer = model_convnext.features[-1]
    
grad_cam = GradCAM(model_convnext, target_layer)

def generate_heatmap(image_path, cam_mask, output_path):
    """Saves a pure RGBA heatmap PNG (no original blended in).
    The frontend slider controls the blend opacity in CSS."""
    original_img = cv2.imread(image_path)
    if original_img is None:
        return None

    h, w = original_img.shape[:2]

    # Resize mask to match original image FIRST
    cam_resized = cv2.resize(cam_mask, (w, h))

    # Smooth the resized CAM mask with Gaussian blur
    cam_smooth = cv2.GaussianBlur(cam_resized, (51, 51), 0)

    # Re-normalize just in case blur lowered the peak
    if np.max(cam_smooth) > 0:
        cam_smooth = cam_smooth / np.max(cam_smooth)

    # Build a JET colormap from the CAM
    heatmap_uint8 = np.uint8(255 * cam_smooth)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2BGRA)

    # Alpha channel: proportional to intensity, so low-activation areas are transparent
    # Make high activation areas more opaque (up to 200 alpha out of 255)
    # Using a smaller power curve (0.8) boosts the mid-tones so the heatmap covers more of the affected area
    alpha = np.uint8((cam_smooth ** 0.8) * 220)
    heatmap_color[:, :, 3] = alpha

    cv2.imwrite(output_path, heatmap_color)
    return output_path

# ============================================================================
# HEALTHY SKIN DETECTION THRESHOLDS
# ============================================================================
# If confidence is below this, the image is likely healthy or unrelated
CONFIDENCE_THRESHOLD = 40.0  # percent — below this = likely healthy skin

# Shannon entropy threshold: if probabilities are spread uniformly across
# all classes the model is "confused" = likely not a disease image
# Max entropy for 23 classes = ln(23) ≈ 3.135
ENTROPY_THRESHOLD = 2.8  # above this = model is too uncertain = healthy / OOD

# ============================================================================
# MAIN PREDICTION LOGIC
# ============================================================================
def get_severity_label(score):
    if score >= 8: return "Critical"
    if score >= 6: return "High"
    if score >= 4: return "Moderate"
    return "Low"


def is_healthy_or_ood(final_probs, confidence):
    """
    Checks two conditions to decide if this image is healthy skin or unrelated:
    1. Confidence is too low (model is not sure about any disease)
    2. Shannon entropy is too high (probabilities are spread out = model confused)
    Returns (is_healthy: bool, reason: str)
    """
    import math
    probs_np = final_probs.detach().cpu().numpy()[0]
    
    # Compute Shannon entropy
    entropy = -sum(p * math.log(p + 1e-9) for p in probs_np)
    
    print(f"[HealthCheck] Confidence: {confidence:.1f}%  Entropy: {entropy:.3f}")
    
    if confidence < CONFIDENCE_THRESHOLD:
        return True, f"low_confidence ({confidence:.1f}%)"
    if entropy > ENTROPY_THRESHOLD:
        return True, f"high_entropy ({entropy:.3f})"
    return False, ""


def run_prediction(image_path):
    """
    Runs the image through the model(s), generates a heatmap, and calculates severity.
    Returns a dict with all results. Includes a 'healthy' flag when no disease is detected.
    """
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    # Need requires_grad for Grad-CAM
    input_tensor.requires_grad = True
    
    # 1. Forward pass (ConvNeXt)
    output1 = model_convnext(input_tensor)
    prob1 = F.softmax(output1, dim=1)
    
    # 2. Ensemble (optional)
    if MODE == "ensemble" and model_effnet is not None:
        with torch.no_grad():
            output2 = model_effnet(input_tensor)
            prob2 = F.softmax(output2, dim=1)
        # Average probabilities
        final_probs = (prob1 + prob2) / 2.0
    else:
        final_probs = prob1
        
    # 3. Get Top Prediction
    confidence, pred_idx = torch.max(final_probs, 1)
    confidence = confidence.item() * 100
    class_idx = pred_idx.item()
    disease_name = CLASSES[class_idx]

    # 4. Healthy / Out-of-distribution check
    healthy, reason = is_healthy_or_ood(final_probs, confidence)

    # 5. Generate Heatmap (using ConvNeXt) — always generate so frontend has an image
    cam_mask = grad_cam(input_tensor, class_idx)

    filename = os.path.basename(image_path)
    heatmap_filename = f"heatmap_{os.path.splitext(filename)[0]}.png"
    heatmap_path = os.path.join("static", "heatmaps", heatmap_filename)
    os.makedirs(os.path.join("static", "heatmaps"), exist_ok=True)
    generate_heatmap(image_path, cam_mask, heatmap_path)

    # 6. If healthy — return early with a special result
    if healthy:
        return {
            "healthy": True,
            "disease": "Healthy Skin",
            "confidence": round(confidence, 1),
            "severity": 0,
            "severity_label": "None",
            "heatmap_path": heatmap_path.replace("\\", "/"),
            "original_path": image_path.replace("\\", "/"),
            "reason": reason
        }

    # 7. Calculate Severity for detected disease
    base_risk = SEVERITY_MAPPING.get(disease_name, 5)
    conf_weight = 0.5 + ((confidence / 100) / 2)
    severity_score = min(10, round(base_risk * conf_weight, 1))
    severity_label = get_severity_label(severity_score)

    return {
        "healthy": False,
        "disease": disease_name,
        "confidence": round(confidence, 1),
        "severity": severity_score,
        "severity_label": severity_label,
        "heatmap_path": heatmap_path.replace("\\", "/"),
        "original_path": image_path.replace("\\", "/")
    }

if __name__ == "__main__":
    # Test script if run directly
    print("Testing prediction on a dummy image if one exists...")
    test_img = "dermatology_dataset/test/Acne and Rosacea Photos/07Rhinophyma1.jpg"
    if os.path.exists(test_img):
        res = run_prediction(test_img)
        print("\n--- Result ---")
        print(json.dumps(res, indent=2))
        print("Heatmap generated at:", res['heatmap_path'])
    else:
        print(f"Could not find test image at {test_img}")
