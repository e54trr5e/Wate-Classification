"""Plot training curves from the current Ultralytics results.csv.

This script intentionally does not create a cross-model accuracy chart.
A fair comparison requires every model to use the same fixed test images
and evaluation protocol. See final_model_comparison.py and the README.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Ultralytics may place the run under either path depending on version/config.
CANDIDATE_CSV_PATHS = [
    BASE_DIR / "runs" / "classify" / "runs" / "yolo_classification"
    / "waste_yolov8n_cls" / "results.csv",
    BASE_DIR / "runs" / "yolo_classification"
    / "waste_yolov8n_cls" / "results.csv",
]

CSV_PATH = next((path for path in CANDIDATE_CSV_PATHS if path.exists()), None)


def main():
    print("\n========== YOLO TRAINING CURVES ==========\n")

    if CSV_PATH is None:
        print("YOLO results.csv not found. Train the YOLO classification model first.")
        return

    with CSV_PATH.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = reader.fieldnames or []

    if not rows:
        print("The YOLO results.csv file is empty.")
        return

    loss_columns = [column for column in columns if "loss" in column.lower()]
    accuracy_columns = [
        column for column in columns
        if "top1" in column.lower() or "top5" in column.lower()
    ]

    for selected_columns, title, ylabel, output_name in [
        (loss_columns, "YOLOv8 Training and Validation Loss", "Loss", "yolo_training_loss.png"),
        (accuracy_columns, "YOLOv8 Classification Training Metrics", "Metric", "yolo_training_metrics.png"),
    ]:
        if not selected_columns:
            print(f"No matching columns found for {title}.")
            continue

        plt.figure(figsize=(9, 5))
        for column in selected_columns:
            values = []
            for row in rows:
                try:
                    values.append(float(row[column]))
                except (TypeError, ValueError):
                    values.append(float("nan"))
            plt.plot(range(1, len(values) + 1), values, label=column)

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        output_path = RESULTS_DIR / output_name
        plt.savefig(output_path, dpi=300)
        plt.close()
        print("Saved:", output_path)

    print("\nNo cross-model accuracy chart was generated. Use a common test split first.")


if __name__ == "__main__":
    main()
