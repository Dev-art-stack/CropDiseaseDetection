import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

from torchvision import models, transforms, datasets
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------

MODEL_PATH = "best_mobilenetv3_plantvillage.pth"
DATA_DIR   = "PlantVillage_Train"

IMAGE_PATH = r"C:\Users\devar\Downloads\WhatsApp Image 2026-04-18 at 12.43.12 PM.jpeg"

IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MC_SAMPLES = 30

print("Using device:", DEVICE)

# -----------------------------
# LOAD CLASS NAMES
# -----------------------------

dataset     = datasets.ImageFolder(DATA_DIR)
class_names = dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)

# -----------------------------
# TRANSFORM
# Better preprocessing: center crop removes background clutter
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# LOAD MODEL
# -----------------------------

model = models.mobilenet_v3_large(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    num_classes
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.to(DEVICE)
model.eval()

# -----------------------------
# LOAD IMAGE
# -----------------------------

image        = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(DEVICE)

# Keep cropped image for display so it matches what model sees
display_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
])
display_image = np.array(display_transform(image))

# -----------------------------
# MC DROPOUT
# -----------------------------

def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def mc_dropout_prediction(model, input_tensor):

    model.eval()
    enable_dropout(model)

    predictions = []

    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            outputs = model(input_tensor)
            probs   = torch.softmax(outputs, dim=1)
            predictions.append(probs.cpu().numpy())

    predictions     = np.vstack(predictions)
    mean_probs      = predictions.mean(axis=0)
    variance        = predictions.var(axis=0)
    predicted_class = np.argmax(mean_probs)
    confidence      = np.max(mean_probs)
    uncertainty     = np.mean(variance)

    return predicted_class, confidence, uncertainty

# -----------------------------
# GRAD-CAM SETUP
# Use second-to-last feature block for better spatial resolution
# -----------------------------

features  = None
gradients = None

def forward_hook(module, input, output):
    global features
    features = output.detach()

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0].detach()

target_layer = model.features[-2]   # Better spatial detail than features[-1]

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(
    lambda m, gi, go: backward_hook(m, gi, go)
)

# -----------------------------
# RUN MC DROPOUT
# -----------------------------

pred_class, confidence, uncertainty = mc_dropout_prediction(
    model, input_tensor
)

print("\nPrediction Results:")
print("Predicted Class  :", class_names[pred_class])
print(f"Confidence       : {confidence * 100:.2f}%")
print(f"Uncertainty      : {uncertainty:.6f}")

# Print top 3 for context
with torch.no_grad():
    model.eval()
    out       = model(input_tensor)
    probs_all = torch.softmax(out, dim=1)[0]
    top3_vals, top3_idx = torch.topk(probs_all, 3)

print("\nTop 3 Predictions:")
for i in range(3):
    print(f"  {class_names[top3_idx[i]]}: {top3_vals[i]*100:.2f}%")

# -----------------------------
# GRAD-CAM GENERATION
# -----------------------------

model.eval()
model.zero_grad()

output      = model(input_tensor)
class_score = output[0, pred_class]
class_score.backward()

pooled_gradients  = torch.mean(gradients, dim=[0, 2, 3])
weighted_features = features.clone()

for i in range(weighted_features.shape[1]):
    weighted_features[:, i, :, :] *= pooled_gradients[i]

heatmap = torch.mean(weighted_features, dim=1).squeeze()
heatmap = torch.relu(heatmap)   # Only keep positive activations
heatmap = heatmap.cpu().numpy()
heatmap /= (np.max(heatmap) + 1e-8)

# -----------------------------
# OVERLAY HEATMAP
# -----------------------------

h, w = display_image.shape[:2]

heatmap_resized = cv2.resize(heatmap, (w, h))
heatmap_uint8   = np.uint8(255 * heatmap_resized)
heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

overlay = cv2.addWeighted(
    display_image,
    0.6,
    heatmap_colored,
    0.4,
    0
)

# -----------------------------
# DISPLAY
# -----------------------------

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

title_color = "darkgreen" if "healthy" in class_names[pred_class] else "darkred"

fig.suptitle(
    f"Predicted: {class_names[pred_class]}  |  "
    f"Confidence: {confidence*100:.2f}%  |  "
    f"Uncertainty: {uncertainty:.6f}",
    fontsize=13,
    fontweight="bold",
    color=title_color
)

axes[0].set_title("Original Image (Cropped)")
axes[0].imshow(display_image)
axes[0].axis("off")

axes[1].set_title("Grad-CAM Heatmap\n(Red = Model focused here)")
axes[1].imshow(heatmap_colored)
axes[1].axis("off")

axes[2].set_title("Overlay")
axes[2].imshow(overlay)
axes[2].axis("off")

plt.tight_layout()
plt.show()