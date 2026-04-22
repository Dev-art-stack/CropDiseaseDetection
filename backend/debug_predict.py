import torch
import torchvision.transforms as transforms
from PIL import Image

# Load model
model = torch.jit.load("model_mobile.pt")
model.eval()

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

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# Test these one by one by changing the filename
image = Image.open("tomato_test.jpg").convert("RGB")

image = transform(image).unsqueeze(0)

with torch.no_grad():

    output = model(image)

    if isinstance(output, tuple):
        output = output[0]

    probs = torch.softmax(output, dim=1)

    print("\nClass Probabilities:\n")

    for i, p in enumerate(probs[0]):
        print(classes[i], ":", float(p))

    pred = torch.argmax(probs, dim=1)

print("\nPredicted:", classes[pred.item()])