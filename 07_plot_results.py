
import csv
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CSV_PATH = (
    BASE_DIR
    / "runs"
    / "classify"
    / "runs"
    / "yolo_classification"
    / "waste_yolov8n_cls"
    / "results.csv"
)

print("\n========== WASTE CLASSIFICATION GRAPHS ==========\n")

# --------------------------------------------------
# 2. READ YOLO CSV WITHOUT PANDAS
# --------------------------------------------------

if CSV_PATH.exists():

    with open(CSV_PATH, "r", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if rows:

        columns = reader.fieldnames
        print("Available metrics:", columns)

        # Find loss and accuracy columns
        loss_columns = [
            c for c in columns
            if "loss" in c.lower()
        ]

        accuracy_columns = [
            c for c in columns
            if "accuracy" in c.lower()
        ]

        # Plot training and validation loss
        if loss_columns:

            plt.figure(figsize=(9, 5))

            for column in loss_columns:
                values = [
                    float(row[column])
                    for row in rows
                ]

                plt.plot(
                    range(1, len(values) + 1),
                    values,
                    label=column
                )

            plt.title("YOLOv8 Training and Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            path = RESULTS_DIR / "yolo_training_loss.png"
            plt.savefig(path, dpi=300)
            plt.show()

            print("Saved:", path)

        # Plot accuracy
        if accuracy_columns:

            plt.figure(figsize=(9, 5))

            for column in accuracy_columns:
                values = [
                    float(row[column])
                    for row in rows
                ]

                plt.plot(
                    range(1, len(values) + 1),
                    values,
                    label=column
                )

            plt.title("YOLOv8 Classification Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            path = RESULTS_DIR / "yolo_training_accuracy.png"
            plt.savefig(path, dpi=300)
            plt.show()

            print("Saved:", path)

    else:
        print("The YOLO CSV file is empty.")

else:
    print("YOLO results.csv not found.")
    print("Skipping training curves.")

# --------------------------------------------------
# 3. MODEL ACCURACY COMPARISON
# --------------------------------------------------

models = [
    "SVM",
    "YOLOv8 Classification"
]

accuracies = [
    80.24,
    91.15
]

plt.figure(figsize=(9, 5))

bars = plt.bar(models, accuracies)

plt.title("Waste Classification Model Accuracy")
plt.xlabel("Models")
plt.ylabel("Test Accuracy (%)")
plt.ylim(0, 100)

for bar, accuracy in zip(bars, accuracies):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center"
    )

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

comparison_path = (
    RESULTS_DIR / "model_accuracy_comparison.png"
)

plt.savefig(comparison_path, dpi=300)
plt.show()

print("\nSaved:", comparison_path)

print("\n========== GRAPH GENERATION COMPLETED ==========")
