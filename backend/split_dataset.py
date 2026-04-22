import os
import shutil
import random

SOURCE_DIR = "PlantVillage"

TRAIN_DIR = "PlantVillage_Train"
TEST_DIR = "PlantVillage_Test"

SPLIT_RATIO = 0.8

random.seed(42)

# Create directories
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

classes = os.listdir(SOURCE_DIR)

for cls in classes:

    src_class = os.path.join(SOURCE_DIR, cls)

    train_class = os.path.join(TRAIN_DIR, cls)
    test_class = os.path.join(TEST_DIR, cls)

    os.makedirs(train_class, exist_ok=True)
    os.makedirs(test_class, exist_ok=True)

    images = os.listdir(src_class)

    random.shuffle(images)

    split_index = int(len(images) * SPLIT_RATIO)

    train_images = images[:split_index]
    test_images = images[split_index:]

    # Copy training images
    for img in train_images:

        src_path = os.path.join(src_class, img)
        dst_path = os.path.join(train_class, img)

        shutil.copy(src_path, dst_path)

    # Copy testing images
    for img in test_images:

        src_path = os.path.join(src_class, img)
        dst_path = os.path.join(test_class, img)

        shutil.copy(src_path, dst_path)

print("Dataset split complete.")