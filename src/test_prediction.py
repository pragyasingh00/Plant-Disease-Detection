
from pathlib import Path
from PIL import Image

from predict import load_prediction_model, predict_image


# --------------------------------
# 1. Select an actual test image
# --------------------------------

TEST_DIR = Path("dataset/processed_900/test")

image_extensions = {".jpg", ".jpeg", ".png"}

image_path = None
actual_class = None

for class_folder in sorted(TEST_DIR.iterdir()):

    if not class_folder.is_dir():
        continue

    images = [
        file for file in class_folder.iterdir()
        if file.suffix.lower() in image_extensions
    ]

    if images:
        image_path = images[0]
        actual_class = class_folder.name
        break


if image_path is None:
    raise FileNotFoundError("No test images found!")


# --------------------------------
# 2. Load trained model
# --------------------------------

print("\nLoading MobileNetV2 model...")

model = load_prediction_model()


# --------------------------------
# 3. Open test image
# --------------------------------

print("\nTesting image:", image_path.name)

image = Image.open(image_path)


# --------------------------------
# 4. Make prediction
# --------------------------------

result = predict_image(image, model)


# --------------------------------
# 5. Display results
# --------------------------------

print("\n" + "=" * 45)

print("PLANT DISEASE PREDICTION RESULT")

print("=" * 45)

print("\nActual Class:", actual_class)

print("Predicted Class:", result["predicted_class"])

print(
    "Confidence Score:",
    round(result["confidence"], 2),
    "%"
)

if actual_class == result["predicted_class"]:
    print("\nResult: CORRECT PREDICTION")
else:
    print("\nResult: INCORRECT PREDICTION")

print("\nPrediction test completed!")
