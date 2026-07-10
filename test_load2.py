import torch
from torchvision.models import efficientnet_v2_s, convnext_tiny
import torch.nn as nn

print("\n--- Testing V6 as EffNet ---")
try:
    ckpt = torch.load('Checkpoints/V6_S3_best.pth', map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    model = efficientnet_v2_s(weights=None)
    in_f = model.classifier[1].in_features
    # Try simple head
    model.classifier[1] = nn.Linear(in_f, 23)
    try:
        model.load_state_dict(state_dict)
        print("V6 matches EffNet Simple Head")
    except Exception as e:
        print("Failed EffNet Simple Head:", e)
        
    # Try complex head
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_f, 512),
        nn.GELU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, 23)
    )
    try:
        model.load_state_dict(state_dict)
        print("V6 matches EffNet Complex Head")
    except Exception as e:
        pass
except Exception as e:
    print(e)
    
print("\n--- Testing V3_1 as ConvNeXt Tiny ---")
try:
    ckpt = torch.load('Checkpoints/V3_1_Stage2_best.pth', map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    model = convnext_tiny(weights=None)
    in_f = model.classifier[2].in_features
    # Try complex head
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_f, 512),
        nn.GELU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, 23)
    )
    try:
        model.load_state_dict(state_dict)
        print("V3_1 matches ConvNeXt Tiny Complex Head")
    except Exception as e:
        # Try simple head
        model.classifier[2] = nn.Linear(in_f, 23)
        try:
            model.load_state_dict(state_dict)
            print("V3_1 matches ConvNeXt Tiny Simple Head")
        except Exception as e:
            print("Failed ConvNeXt Tiny:", e)
except Exception as e:
    print(e)
