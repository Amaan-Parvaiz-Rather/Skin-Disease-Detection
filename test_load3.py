import torch
from torchvision.models import convnext_base
import torch.nn as nn

try:
    ckpt = torch.load('Checkpoints/V5_S2_best.pth', map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    model = convnext_base(weights=None)
    in_f = model.classifier[2].in_features
    # Try simple head
    model.classifier[2] = nn.Linear(in_f, 23)
    try:
        model.load_state_dict(state_dict)
        print("V5 matches ConvNeXt Base Simple Head")
    except Exception as e:
        print("Failed ConvNeXt Base Simple Head:", e)
except Exception as e:
    print(e)
