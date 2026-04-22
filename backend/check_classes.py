from torchvision import datasets

dataset = datasets.ImageFolder("PlantVillage")

print("Number of classes:", len(dataset.classes))

print("\nClass names:")
for i, c in enumerate(dataset.classes):
    print(i, ":", c)