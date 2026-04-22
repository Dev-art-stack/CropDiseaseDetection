import torch
import torch.nn as nn

from torchvision import models, datasets

# -----------------------------
# CONFIG
# -----------------------------

MODEL_PATH = "best_mobilenetv3_plantvillage.pth"
EXPORT_PATH = "model_mobile.pt"

DEVICE = torch.device("cpu")

# -----------------------------
# LOAD DATASET (for class count)
# -----------------------------

dataset = datasets.ImageFolder("PlantVillage")

num_classes = len(dataset.classes)

print("Number of classes:", num_classes)

# -----------------------------
# BUILD MODEL (DEFAULT STRUCTURE)
# -----------------------------

model = models.mobilenet_v3_large(
    weights=None
)

# IMPORTANT:
# Only replace last layer
# NOT full classifier

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    num_classes
)

# -----------------------------
# LOAD TRAINED WEIGHTS
# -----------------------------

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

# -----------------------------
# EXPORT MODEL
# -----------------------------

dummy_input = torch.randn(
    1,
    3,
    224,
    224
)

scripted_model = torch.jit.trace(
    model,
    dummy_input
)

torch.jit.save(
    scripted_model,
    EXPORT_PATH
)

print("\n✅ Model exported successfully!")
print("Saved as:", EXPORT_PATH)