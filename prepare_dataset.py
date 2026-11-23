
from datasets import load_dataset
from collections import defaultdict
from pathlib import Path
import random

# -----------------------------
# 1. Configuration
# -----------------------------

random_seed = 42
random.seed(random_seed)

output_dir = Path("dataset/processed_900")

classes = {
    "Tomato healthy": "Tomato_Healthy",
    "Tomato Early blight": "Tomato_Early_Blight",
    "Tomato Late blight": "Tomato_Late_Blight",
    "Tomato Bacterial spot": "Tomato_Bacterial_Spot",
    "Potato healthy": "Potato_Healthy",
    "Potato Early blight": "Potato_Early_Blight",
    "Potato Late blight": "Potato_Late_Blight",
    "Pepper bell healthy": "Pepper_Healthy",
    "Pepper bell Bacterial spot": "Pepper_Bacterial_Spot"
}

# -----------------------------
# 2. Load dataset
# -----------------------------

print("Loading PlantVillage dataset...")

dataset = load_dataset(
    "manoela/plantvillage",
    split="train"
)

print("Total available images:", len(dataset))

# -----------------------------
# 3. Collect image indices
# -----------------------------

class_indices = defaultdict(list)

for index, caption in enumerate(dataset["caption"]):
    if caption in classes:
        class_indices[caption].append(index)

print("\nAvailable images per selected class:")

for caption in classes:
    print(caption, ":", len(class_indices[caption]))

# -----------------------------
# 4. Select 100 images per class
# -----------------------------

selected_images = {}

for caption, folder_name in classes.items():

    available = class_indices[caption]

    if len(available) < 100:
        raise ValueError(
            f"Not enough images for {caption}"
        )

    selected_images[caption] = random.sample(
        available, 100
    )

print("\nSelected 100 images from each class.")

# -----------------------------
# 5. Create train/validation/test
# -----------------------------

splits = {
    "train": [],
    "validation": [],
    "test": []
}

for caption, indices in selected_images.items():

    random.shuffle(indices)

    splits["train"].extend(
        [(caption, index) for index in indices[:70]]
    )

    splits["validation"].extend(
        [(caption, index) for index in indices[70:85]]
    )

    splits["test"].extend(
        [(caption, index) for index in indices[85:100]]
    )

# -----------------------------
# 6. Save images into folders
# -----------------------------

print("\nSaving selected images...")

for split_name, items in splits.items():

    for caption, index in items:

        class_folder = classes[caption]

        save_folder = (
            output_dir / split_name / class_folder
        )

        save_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        image = dataset[index]["image"]

        image = image.convert("RGB")

        filename = f"image_{index}.jpg"

        image.save(save_folder / filename, quality=95)

# -----------------------------
# 7. Verify dataset
# -----------------------------

print("\nDATASET PREPARATION COMPLETED")
print("--------------------------------")

for split_name in splits:

    print(f"\n{split_name.upper()}")

    total = 0

    for caption, folder_name in classes.items():

        folder = output_dir / split_name / folder_name

        count = len(list(folder.glob("*.jpg")))

        total += count

        print(f"{folder_name}: {count}")

    print("Total:", total)

print("\nAll 900 images have been organized.")
