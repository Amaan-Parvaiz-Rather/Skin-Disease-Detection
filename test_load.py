import torch
from torchvision.models import convnext_base
import torch.nn as nn

try:
    ckpt = torch.load('Checkpoints/V5_S2_best.pth', map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    model_convnext = convnext_base(weights=None)
    in_features = model_convnext.classifier[2].in_features
    model_convnext.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.GELU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, 23)
    )
    
    model_convnext.load_state_dict(state_dict)
    print("V5_S2_best.pth loaded successfully as convnext_base")
except Exception as e:
    print(f"Error loading V5_S2_best.pth: {e}")

try:
    ckpt = torch.load('Checkpoints/V6_S3_best.pth', map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    print("V6_S3_best.pth state dict keys:", list(state_dict.keys())[:5])
except Exception as e:
    pass
