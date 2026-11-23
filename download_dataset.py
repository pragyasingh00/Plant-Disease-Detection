
from datasets import load_dataset
from collections import Counter

print("Loading PlantVillage dataset...")

dataset = load_dataset(
    "manoela/plantvillage",
    split="train"
)

print("\nFirst 10 captions:\n")

for i in range(10):
    print(f"Image {i + 1}: {dataset[i]['caption']}")

print("\nMost common captions:\n")

caption_counts = Counter(dataset["caption"])

for caption, count in caption_counts.most_common(30):
    print(f"{caption}: {count}")

print("\nTotal unique captions:", len(caption_counts))
