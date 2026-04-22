import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

# ------------------
# CONFIG
# ------------------

TEST_DIR = "PlantVillage_Test"

MODEL_PATH = "best_mobilenetv3_plantvillage.pth"

IMG_SIZE = 224
BATCH_SIZE = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# ------------------
# TRANSFORMS
# ------------------

test_transform = transforms.Compose([

    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------
# DATASET
# ------------------

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

num_classes = len(test_dataset.classes)

print("Number of classes:", num_classes)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ------------------
# LOAD MODEL
# ------------------

model = models.mobilenet_v3_large(
    weights=None
)

# SAME classifier structure as train.py

model.classifier = nn.Sequential(

    nn.Linear(
        model.classifier[0].in_features,
        1024
    ),

    nn.Hardswish(),

    nn.Dropout(p=0.3),

    nn.Linear(
        1024,
        num_classes
    )
)

print("Loading model weights...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully!")

# ------------------
# PREDICTION LOOP
# ------------------

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in tqdm(test_loader):

        images = images.to(DEVICE)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        y_true.extend(
            labels.numpy()
        )

        y_pred.extend(
            preds.cpu().numpy()
        )

# ------------------
# CONFUSION MATRIX
# ------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=test_dataset.classes,
    yticklabels=test_dataset.classes
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

print("\nConfusion matrix saved as confusion_matrix.png")

# ------------------
# CLASSIFICATION REPORT
# ------------------

print("\nClassification Report:\n")

print(

    classification_report(
        y_true,
        y_pred,
        target_names=test_dataset.classes
    )

)