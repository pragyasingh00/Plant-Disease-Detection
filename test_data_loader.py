
from src.data_loader import get_datasets

print("\nTesting CNN data loader...")

train_ds, val_ds, test_ds, class_names = get_datasets(
    model_type="cnn"
)

print("\nClass names:")
print(class_names)

for images, labels in train_ds.take(1):
    print("\nImage batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Minimum pixel value:", images.numpy().min())
    print("Maximum pixel value:", images.numpy().max())

print("\nData loader test completed successfully!")
