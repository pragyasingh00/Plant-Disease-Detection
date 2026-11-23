
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from data_loader import get_datasets


# --------------------------------
# 1. Load test dataset
# --------------------------------

print("\nLoading test dataset...")

_, _, test_ds, class_names = get_datasets(
    model_type="cnn"
)


# --------------------------------
# 2. Load best trained CNN model
# --------------------------------

print("\nLoading trained CNN model...")

model = tf.keras.models.load_model(
    "models/cnn_best.keras"
)


# --------------------------------
# 3. Evaluate model
# --------------------------------

print("\nEvaluating CNN model...")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print("\nTest Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")


# --------------------------------
# 4. Generate predictions
# --------------------------------

print("\nGenerating predictions...")

predictions = model.predict(test_ds)

y_pred = np.argmax(predictions, axis=1)

y_true = np.concatenate([
    np.argmax(labels.numpy(), axis=1)
    for images, labels in test_ds
])


# --------------------------------
# 5. Calculate evaluation metrics
# --------------------------------

accuracy = accuracy_score(y_true, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print(report)


# --------------------------------
# 6. Create results folder
# --------------------------------

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)


# --------------------------------
# 7. Save classification report
# --------------------------------

with open(
    results_dir / "cnn_classification_report.txt",
    "w"
) as file:

    file.write("CNN MODEL EVALUATION\n\n")

    file.write(
        f"Test Accuracy: {accuracy * 100:.2f}%\n\n"
    )

    file.write(report)


# --------------------------------
# 8. Generate confusion matrix
# --------------------------------

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 9))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("CNN Confusion Matrix")

plt.xlabel("Predicted Class")

plt.ylabel("Actual Class")

plt.xticks(rotation=45, ha="right")

plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    results_dir / "cnn_confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nEvaluation completed successfully!")

print("Classification report saved in results folder.")

print("Confusion matrix saved in results folder.")
