import torch
from PIL import Image
from torchvision import transforms, datasets

# -----------------------------
# CONFIG
# -----------------------------

MODEL_PATH = "model_mobile.pt"
IMAGE_PATH = "test_leaf.JPG"   # change if needed
DATA_DIR = "PlantVillage"

DEVICE = torch.device("cpu")

# -----------------------------
# LOAD CLASS NAMES
# -----------------------------

dataset = datasets.ImageFolder(DATA_DIR)
class_names = dataset.classes

# -----------------------------
# LOAD MODEL
# -----------------------------

model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# -----------------------------
# TRANSFORM
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# LOAD IMAGE
# -----------------------------

image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(DEVICE)

# -----------------------------
# PREDICTION
# -----------------------------

with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    confidence, pred_class = torch.max(probs, 1)

predicted_label = class_names[pred_class.item()]

# -----------------------------
# DECISION LOGIC
# -----------------------------

CONF_THRESHOLD = 0.80

if confidence.item() < CONF_THRESHOLD:

    print("⚠️ Warning: Low confidence prediction")

else:

    print("✅ Prediction is reliable")

CLASS_MAP = {
"Pepper__bell___Bacterial_spot": "Bell Pepper - Bacterial Spot",
"Pepper__bell___healthy": "Bell Pepper - Healthy",
"Potato___Early_blight": "Potato - Early Blight",
"Potato___Late_blight": "Potato - Late Blight",
"Potato___healthy": "Potato - Healthy",
"Tomato_Early_blight": "Tomato - Early Blight",
"Tomato_Late_blight": "Tomato - Late Blight",
"Tomato_Septoria_leaf_spot": "Tomato - Septoria Leaf Spot",
"Tomato_healthy": "Tomato - Healthy"
}

display_name = CLASS_MAP[predicted_label]

print("Predicted Disease:", display_name)

# -----------------------------
# OUTPUT
# -----------------------------

print("\n📱 Mobile Inference Result:")
print("Predicted Class:", predicted_label)
print("Confidence: {:.2f}%".format(confidence.item() * 100))