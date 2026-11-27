"""Display saved model metrics without implying an unfair benchmark.

The current ML, CNN and YOLO scripts do not yet share one identical test split.
This script therefore prints the evaluation protocol for each result but does
not generate a cross-model ranking or comparison chart.
"""

import json
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"


def load_pickle(path):
    if not path.exists():
        return None
    with path.open("rb") as file:
        return pickle.load(file)


def load_json(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def percent(value):
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def main():
    ml_results = load_pickle(RESULTS_DIR / "ml_model_results.pkl")
    cnn_results = load_json(RESULTS_DIR / "cnn_results.json")
    yolo_results = load_json(RESULTS_DIR / "yolo_classification_results.json")

    print("\nSAVED WASTE CLASSIFICATION RESULTS")
    print("=" * 78)
    print("These metrics come from different evaluation protocols and are NOT")
    print("a fair head-to-head comparison until all models use one fixed test set.")
    print("-" * 78)

    if ml_results:
        print("\nTraditional ML (current script: stratified 80/20 test split)")
        for result in ml_results:
            print(
                f"{result.get('name', 'ML model'):<28}"
                f"Accuracy: {percent(result.get('accuracy')):<12}"
                f"Macro F1: {percent(result.get('macro_f1'))}"
            )
    else:
        print("\nTraditional ML results not found.")

    if cnn_results:
        print("\nCustom CNN (current script: validation split)")
        print("Accuracy:", percent(cnn_results.get("accuracy")))
        print("Weighted F1:", percent(cnn_results.get("f1_score")))
    else:
        print("\nCNN results not found.")

    if yolo_results:
        print("\nYOLOv8 classification (current script: held-out test folder)")
        print("Accuracy:", percent(yolo_results.get("accuracy")))
        print("Macro F1:", percent(yolo_results.get("macro_f1")))
        print("Weighted F1:", percent(yolo_results.get("weighted_f1")))
    else:
        print("\nYOLO results not found.")

    print("\nNo combined ranking or comparison graph was generated.")
    print("Next step: migrate all model families to the same saved image split.")


if __name__ == "__main__":
    main()
