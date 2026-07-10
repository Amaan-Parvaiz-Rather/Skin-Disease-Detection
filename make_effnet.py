import re
import os

with open("train_image_v3.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Imports
code = code.replace(
    "from torchvision.models import convnext_base, ConvNeXt_Base_Weights",
    "from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights"
)

# 2. Model initialisation
code = code.replace(
    "weights = ConvNeXt_Base_Weights.IMAGENET1K_V1",
    "weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1"
)
code = code.replace(
    "model   = convnext_base(weights=weights)",
    "model   = efficientnet_v2_s(weights=weights)"
)

# 3. Classifier access
# In ConvNeXt it is model.classifier[2]. In EfficientNetV2 it is model.classifier[1].
code = code.replace(
    "in_features = model.classifier[2].in_features",
    "in_features = model.classifier[1].in_features"
)
code = code.replace(
    "model.classifier[2] = nn.Sequential(",
    "model.classifier[1] = nn.Sequential("
)

# 4. Save paths
code = code.replace('"V3_S1_best.pth"', '"EffNet_S1_best.pth"')
code = code.replace('"V3_S1_last.pth"', '"EffNet_S1_last.pth"')
code = code.replace('"V3_S2_best.pth"', '"EffNet_S2_best.pth"')
code = code.replace('"V3_S2_last.pth"', '"EffNet_S2_last.pth"')
code = code.replace('"V3_S3_best.pth"', '"EffNet_S3_best.pth"')
code = code.replace('"V3_S3_last.pth"', '"EffNet_S3_last.pth"')

# 5. Result text
code = code.replace('results_v3.txt', 'results_effnet.txt')
code = code.replace('confusion_matrix_v3.png', 'confusion_matrix_effnet.png')
code = code.replace('V3', 'EffNet')

# 6. Freezing in Stage 1
# V3 uses: for param in model.features.parameters(): param.requires_grad = False
# This works for both ConvNeXt and EfficientNet since both have `model.features`
# No change needed.

with open("train_efficientnet.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Generated train_efficientnet.py")
