
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------
# 1. Create results directory
# --------------------------------

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# --------------------------------
# 2. Model comparison data
# --------------------------------

data = {
    "Model": ["Custom CNN", "MobileNetV2"],
    "Test Accuracy (%)": [68.89, 94.81],
    "Macro Precision": [0.72, 0.95],
    "Macro Recall": [0.69, 0.95],
    "Macro F1-Score": [0.68, 0.95],
    "Total Parameters": [13236041, 2269513]
}

df = pd.DataFrame(data)

# --------------------------------
# 3. Display comparison
# --------------------------------

print("\nMODEL PERFORMANCE COMPARISON")
print("=" * 65)

print(df.to_string(index=False))

# --------------------------------
# 4. Save comparison CSV
# --------------------------------

df.to_csv(
    results_dir / "model_comparison.csv",
    index=False
)

# --------------------------------
# 5. Plot accuracy comparison
# --------------------------------

plt.figure(figsize=(8, 5))

bars = plt.bar(
    df["Model"],
    df["Test Accuracy (%)"]
)

plt.title("CNN vs MobileNetV2 - Test Accuracy")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    results_dir / "accuracy_comparison.png",
    dpi=300
)

plt.close()

# --------------------------------
# 6. Plot macro metrics comparison
# --------------------------------

metrics = [
    "Macro Precision",
    "Macro Recall",
    "Macro F1-Score"
]

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    [i - width / 2 for i in x],
    df.iloc[0][metrics],
    width,
    label="Custom CNN"
)

plt.bar(
    [i + width / 2 for i in x],
    df.iloc[1][metrics],
    width,
    label="MobileNetV2"
)

plt.xticks(list(x), metrics)
plt.ylabel("Score")
plt.title("Macro Metrics Comparison")
plt.ylim(0, 1.1)
plt.legend()
plt.tight_layout()

plt.savefig(
    results_dir / "metrics_comparison.png",
    dpi=300
)

plt.close()

# --------------------------------
# 7. Final message
# --------------------------------

print("\nComparison report generated successfully!")

print("\nFiles saved:")
print("1. results/model_comparison.csv")
print("2. results/accuracy_comparison.png")
print("3. results/metrics_comparison.png")
