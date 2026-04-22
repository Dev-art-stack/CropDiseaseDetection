import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_DIR  = "PlantVillage_Test"
IMG_SIZE = 224

classes    = datasets.ImageFolder(VAL_DIR).classes
num_classes = len(classes)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_loader = DataLoader(
    datasets.ImageFolder(VAL_DIR, transform=transform),
    batch_size=32, shuffle=False
)

train_loader = DataLoader(
    datasets.ImageFolder("PlantVillage_Train", transform=transform),
    batch_size=32,
    shuffle=True
)

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return 100 * correct / total

# ---- Your MobileNetV3 ----
mobilenet = models.mobilenet_v3_large(weights=None)
mobilenet.classifier[3] = nn.Linear(mobilenet.classifier[3].in_features, num_classes)
mobilenet.load_state_dict(torch.load("best_mobilenetv3_plantvillage.pth", map_location=DEVICE, weights_only=True))
mobilenet = mobilenet.to(DEVICE)

# ---- ResNet50 (pretrained on ImageNet, fine-tune head) ----
resnet = models.resnet50(weights="IMAGENET1K_V1")
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
resnet = resnet.to(DEVICE)

# ---- EfficientNet B0 ----
efficientnet = models.efficientnet_b0(weights="IMAGENET1K_V1")
efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, num_classes)
efficientnet = efficientnet.to(DEVICE)

# ---- VGG16 ----
vgg = models.vgg16(weights="IMAGENET1K_V1")
vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, num_classes)
vgg = vgg.to(DEVICE)

results = {
    "MobileNetV3 (Yours)": evaluate(mobilenet, val_loader),
    "ResNet50":            evaluate(resnet,     val_loader),
    "EfficientNet-B0":     evaluate(efficientnet, val_loader),
    "VGG16":               evaluate(vgg,        val_loader),
}

print("\n========== Model Comparison ==========")
for model_name, acc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"{model_name:30s}: {acc:.2f}%")

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def detailed_evaluate(model, loader, class_names):
    model.eval()
    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def fine_tune(model, train_loader, val_loader, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for images, labels in tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1} Val Acc: {acc:.2f}%")

    return evaluate(model, val_loader)

print("Fine-tuning ResNet50...")
resnet_acc = fine_tune(resnet, train_loader, val_loader)

print("Fine-tuning EfficientNet...")
eff_acc = fine_tune(efficientnet, train_loader, val_loader)

print("Fine-tuning VGG16...")
vgg_acc = fine_tune(vgg, train_loader, val_loader)

results = {
    "MobileNetV3": evaluate(mobilenet, val_loader),
    "ResNet50":             resnet_acc,
    "EfficientNet-B0":      eff_acc,
    "VGG16":                vgg_acc,
}

print("\n========== Final Model Comparison ==========")
for model_name, acc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"{model_name:30s}: {acc:.2f}%")