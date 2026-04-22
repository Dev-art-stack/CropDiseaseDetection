from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image

import torch.nn.functional as F
import numpy as np
import cv2
import base64

from torchvision import models
import torch.nn as nn

app = Flask(__name__)

# -----------------------------
# DEFINE CLASSES
# -----------------------------

classes = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Septoria_leaf_spot",
    "Tomato_healthy"
]

# -----------------------------
# HARDCODED DISEASE INFO
# -----------------------------

DISEASE_INFO = {

    "Pepper__bell___Bacterial_spot": {
        "about": "Bacterial spot is a common and destructive disease of pepper plants caused by the bacterium Xanthomonas campestris. It thrives in warm, wet conditions and can severely reduce crop yield.",
        "symptoms": "Small, water-soaked spots appear on leaves that turn brown or black with yellow halos. Infected fruits develop raised, scab-like lesions and leaves may drop prematurely.",
        "remedies": "Apply copper-based bactericides early in the season. Remove and destroy infected plant material immediately. Avoid working with plants when they are wet to prevent spread.",
        "precautions": [
            "Use certified disease-free seeds and transplants",
            "Avoid overhead irrigation — use drip irrigation instead",
            "Rotate crops every 2-3 years",
            "Disinfect gardening tools regularly",
            "Plant in well-drained soil with good air circulation"
        ]
    },

    "Pepper__bell___healthy": {
        "about": "Your bell pepper plant appears to be healthy! No signs of disease have been detected. Healthy pepper plants produce high yields of quality fruits.",
        "symptoms": "No symptoms detected. Leaves are green, firm, and free of spots or discoloration.",
        "remedies": "No treatment needed. Continue your current care routine.",
        "precautions": [
            "Water regularly but avoid overwatering",
            "Ensure at least 6 hours of sunlight daily",
            "Use balanced fertilizer every 2-3 weeks",
            "Monitor regularly for early signs of disease",
            "Keep weeds away from the base of the plant"
        ]
    },

    "Potato___Early_blight": {
        "about": "Early blight is a fungal disease of potato plants caused by Alternaria solani. It is one of the most common potato diseases worldwide and can cause significant yield loss if left untreated.",
        "symptoms": "Dark brown to black spots with concentric rings appear on older leaves first, giving a target-board appearance. Affected leaves turn yellow and drop, and stems may also show lesions.",
        "remedies": "Apply fungicides containing chlorothalonil or mancozeb at the first sign of disease. Remove infected leaves promptly. Ensure adequate fertilization as stressed plants are more susceptible.",
        "precautions": [
            "Plant certified disease-free seed potatoes",
            "Rotate crops — avoid planting potatoes in the same spot for 2-3 years",
            "Maintain proper plant spacing for good air circulation",
            "Avoid wetting foliage during irrigation",
            "Remove and destroy crop debris after harvest"
        ]
    },

    "Potato___Late_blight": {
        "about": "Late blight is a devastating disease caused by the water mold Phytophthora infestans. It is the same disease responsible for the Irish Potato Famine and can destroy an entire crop within days under favorable conditions.",
        "symptoms": "Water-soaked, pale green to brown lesions appear on leaves, often with a white mold growth on the underside. Infected tubers show reddish-brown rot that can spread to the whole potato.",
        "remedies": "Apply fungicides such as metalaxyl or cymoxanil immediately. Destroy infected plants completely — do not compost them. Harvest tubers before disease spreads to the soil.",
        "precautions": [
            "Use resistant potato varieties when available",
            "Avoid planting in low-lying, poorly drained areas",
            "Monitor weather — disease spreads rapidly in cool, wet conditions",
            "Destroy all volunteer potato plants",
            "Store harvested potatoes in cool, dry, well-ventilated conditions"
        ]
    },

    "Potato___healthy": {
        "about": "Your potato plant appears to be healthy! No signs of disease detected. Healthy potato plants produce good yields of quality tubers.",
        "symptoms": "No symptoms detected. Leaves are green and healthy with no spots or discoloration.",
        "remedies": "No treatment needed. Continue your current care routine.",
        "precautions": [
            "Hill soil around the base of plants as they grow",
            "Water consistently — uneven watering causes hollow heart",
            "Fertilize with a potato-specific fertilizer",
            "Monitor for pest activity such as aphids and beetles",
            "Harvest once the tops die back naturally"
        ]
    },

    "Tomato_Early_blight": {
        "about": "Early blight in tomatoes is caused by the fungus Alternaria solani. It is one of the most common tomato diseases and typically affects older, lower leaves first before spreading upward.",
        "symptoms": "Dark brown spots with concentric rings and yellow halos appear on lower leaves. Stems may show dark, sunken lesions. Heavily infected plants lose leaves, exposing fruits to sunscald.",
        "remedies": "Apply copper-based or chlorothalonil fungicides every 7-10 days. Remove affected leaves immediately. Mulch around plants to prevent soil splash onto leaves.",
        "precautions": [
            "Stake or cage plants to improve air circulation",
            "Water at the base — avoid wetting leaves",
            "Rotate tomatoes with non-solanaceous crops",
            "Remove and destroy infected plant debris",
            "Apply mulch to reduce soil splash"
        ]
    },

    "Tomato_Late_blight": {
        "about": "Late blight in tomatoes is caused by Phytophthora infestans, the same pathogen that affects potatoes. It spreads rapidly in cool, moist conditions and can devastate a crop quickly.",
        "symptoms": "Large, irregular, greasy-looking dark brown or black patches appear on leaves and stems. White mold may be visible on undersides of leaves. Fruits develop firm, brown, rough-textured rot.",
        "remedies": "Apply fungicides containing chlorothalonil, mancozeb, or copper hydroxide at first sign. Remove and bag infected plant parts — do not compost. In severe cases, remove entire plants.",
        "precautions": [
            "Plant resistant tomato varieties",
            "Avoid overhead watering",
            "Ensure wide plant spacing for airflow",
            "Monitor forecasts — spray preventively before wet weather",
            "Never save seeds from infected fruits"
        ]
    },

    "Tomato_Septoria_leaf_spot": {
        "about": "Septoria leaf spot is a very common fungal disease of tomatoes caused by Septoria lycopersici. While it rarely kills plants outright, it can severely defoliate them and reduce fruit production.",
        "symptoms": "Small, circular spots with dark brown borders and lighter grey or tan centers appear on lower leaves first. Tiny black dots (fungal spores) are visible in the center of spots. Infected leaves turn yellow and drop.",
        "remedies": "Apply fungicides such as chlorothalonil or copper-based sprays every 7-14 days. Remove infected leaves as soon as spots appear. Avoid working with plants when wet.",
        "precautions": [
            "Remove lower leaves that touch the soil",
            "Mulch heavily to prevent soil splash",
            "Avoid overhead watering",
            "Rotate tomatoes every 2-3 years",
            "Clean up all plant debris at end of season"
        ]
    },

    "Tomato_healthy": {
        "about": "Your tomato plant appears to be healthy! No signs of disease detected. Healthy tomato plants are productive and will yield high-quality fruits with proper care.",
        "symptoms": "No symptoms detected. Leaves are deep green, firm, and free of spots or lesions.",
        "remedies": "No treatment needed. Continue your current care routine.",
        "precautions": [
            "Water deeply and consistently — about 1-2 inches per week",
            "Fertilize with tomato-specific fertilizer every 2 weeks",
            "Prune suckers for better airflow and larger fruits",
            "Support plants with stakes or cages",
            "Inspect regularly for pests like aphids and hornworms"
        ]
    }
}


def get_disease_info(disease_name: str) -> dict:
    """Returns hardcoded disease info for the given disease name."""
    return DISEASE_INFO.get(disease_name, {
        "about":       f"{disease_name.replace('_', ' ')} — please consult a local agricultural expert for details.",
        "symptoms":    "Look for unusual spots, discoloration, or lesions on leaves and stems.",
        "remedies":    "Consult a local agricultural expert for appropriate treatment options.",
        "precautions": [
            "Remove and destroy infected plant parts",
            "Avoid overhead watering",
            "Ensure good air circulation around plants",
            "Rotate crops regularly"
        ]
    })


# -----------------------------
# LOAD MODEL
# -----------------------------

model = models.mobilenet_v3_large(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(classes)
)

model.load_state_dict(
    torch.load(
        "best_mobilenetv3_plantvillage.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

# -----------------------------
# TRANSFORMS
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

display_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
])

# -----------------------------
# LEAF DETECTION
# -----------------------------

def is_leaf(image: Image.Image, threshold: float = 0.08) -> tuple:

    img_np  = np.array(image.resize((224, 224)))
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    lower_green  = np.array([25,  30,  30])
    upper_green  = np.array([95, 255, 255])
    lower_yellow = np.array([15,  30,  30])
    upper_yellow = np.array([25, 255, 255])

    green_mask    = cv2.inRange(img_hsv, lower_green,  upper_green)
    yellow_mask   = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    combined_mask = cv2.bitwise_or(green_mask, yellow_mask)

    total_pixels = 224 * 224
    green_pixels = np.count_nonzero(combined_mask)
    green_ratio  = green_pixels / total_pixels

    if green_ratio < threshold:
        if green_ratio < 0.02:
            message = "No leaf detected. Please upload a clear image of a plant leaf."
        else:
            message = "Image may not contain a plant leaf. Please upload a clearer leaf image."
        return False, green_ratio, message

    return True, green_ratio, "Leaf detected"


# -----------------------------
# MC DROPOUT
# -----------------------------

def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def mc_dropout_prediction(model, input_tensor, n_samples=30):

    model.eval()
    enable_dropout(model)
    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            output = model(input_tensor)
            probs  = F.softmax(output, dim=1)
            predictions.append(probs.cpu().numpy())

    predictions     = np.vstack(predictions)
    mean_probs      = predictions.mean(axis=0)
    variance        = predictions.var(axis=0)
    predicted_class = int(np.argmax(mean_probs))
    confidence      = float(np.max(mean_probs)) * 100
    uncertainty     = float(np.mean(variance))

    return predicted_class, confidence, uncertainty


# -----------------------------
# GRAD-CAM
# -----------------------------

def generate_gradcam(model, input_tensor, class_idx):

    gradients   = []
    activations = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer    = model.features[-2]
    forward_handle  = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(
        lambda m, gi, go: backward_hook(m, gi, go)
    )

    output = model(input_tensor)
    if isinstance(output, tuple):
        output = output[0]

    model.zero_grad()
    output[:, class_idx].backward()

    grads   = gradients[0]
    acts    = activations[0]
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam     = torch.relu(torch.sum(weights * acts, dim=1))

    cam = cam.squeeze().detach().cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    forward_handle.remove()
    backward_handle.remove()

    return cam


# -----------------------------
# PREDICT ROUTE
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    file  = request.files["image"]
    image = Image.open(file).convert("RGB")

    # Step 1: Leaf detection
    leaf_found, green_ratio, leaf_message = is_leaf(image)

    if not leaf_found:
        return jsonify({
            "error":       "no_leaf",
            "message":     leaf_message,
            "green_ratio": round(green_ratio, 4)
        }), 200

    # Step 2: Predict
    input_tensor  = transform(image).unsqueeze(0)
    display_image = np.array(display_transform(image))

    use_uncertainty = request.form.get("uncertainty", "false").lower() == "true"

    if use_uncertainty:
        pred_class, confidence_value, uncertainty_value = mc_dropout_prediction(
            model, input_tensor
        )
    else:
        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            probs            = F.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)
            pred_class       = pred.item()
            confidence_value = confidence.item() * 100
            uncertainty_value = None

    prediction = classes[pred_class]

    # Step 3: Get disease info (hardcoded)
    disease_info = get_disease_info(prediction)

    # Step 4: Grad-CAM overlay
    cam         = generate_gradcam(model, input_tensor, pred_class)
    h, w        = display_image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    cam_uint8   = np.uint8(255 * cam_resized)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    display_bgr = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)

    overlay_bgr = cv2.addWeighted(
        display_bgr, 0.6,
        heatmap_bgr, 0.4,
        0
    )

    _, buffer      = cv2.imencode(".jpg", overlay_bgr)
    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

    response_data = {
        "prediction":   prediction,
        "confidence":   round(confidence_value, 2),
        "heatmap":      heatmap_base64,
        "disease_info": disease_info
    }

    if uncertainty_value is not None:
        response_data["uncertainty"] = round(uncertainty_value, 6)

    return jsonify(response_data)


# -----------------------------
# RUN SERVER
# -----------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )