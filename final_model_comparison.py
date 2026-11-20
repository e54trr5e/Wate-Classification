
import os
import json
import pickle
import matplotlib.pyplot as plt


# ==========================================
# 1. FILE PATHS
# ==========================================

RESULTS_DIR = "./results"

ML_FILE = os.path.join(RESULTS_DIR, "ml_model_results.pkl")
CNN_FILE = os.path.join(RESULTS_DIR, "cnn_results.json")
YOLO_FILE = os.path.join(RESULTS_DIR, "yolo_classification_results.json")


# ==========================================
# 2. LOAD SAVED RESULTS
# ==========================================

def load_file(path, file_type):

    if not os.path.exists(path):
        print(f"WARNING: File not found: {path}")
        return None

    try:
        if file_type == "pickle":
            with open(path, "rb") as file:
                return pickle.load(file)

        elif file_type == "json":
            with open(path, "r") as file:
                return json.load(file)

    except Exception as error:
        print(f"ERROR loading {path}: {error}")

    return None


ml_results = load_file(ML_FILE, "pickle")
cnn_results = load_file(CNN_FILE, "json")
yolo_results = load_file(YOLO_FILE, "json")


# ==========================================
# 3. METRIC HELPER FUNCTIONS
# ==========================================

def normalize_metric(value):

    if value is None:
        return None

    try:
        value = float(value)

        # Convert decimal values into percentages
        if 0 <= value <= 1:
            value *= 100

        return round(value, 2)

    except (TypeError, ValueError):
        return None


def find_metric(data, possible_names):
    """
    Search for a metric inside a dictionary.
    Supports nested JSON structures.
    """

    if isinstance(data, dict):

        for key, value in data.items():

            normalized_key = (
                str(key).lower()
                .replace("_", "")
                .replace("-", "")
                .replace(" ", "")
                .replace("/", "")
            )

            if normalized_key in possible_names:
                result = normalize_metric(value)

                if result is not None:
                    return result

        for value in data.values():

            result = find_metric(value, possible_names)

            if result is not None:
                return result

    elif isinstance(data, list):

        for item in data:

            result = find_metric(item, possible_names)

            if result is not None:
                return result

    return None


# ==========================================
# 4. PREPARE MODEL COMPARISON DATA
# ==========================================

models = []


# ------------------------------------------
# Traditional Machine Learning Models
# ------------------------------------------

if ml_results:

    for result in ml_results:

        models.append({
            "name": result["name"],

            "accuracy": normalize_metric(
                result.get("accuracy")
            ),

            "precision": normalize_metric(
                result.get("precision")
            ),

            "recall": normalize_metric(
                result.get("recall")
            ),

            "f1_score": normalize_metric(
                result.get("f1_score")
            ),

            "evaluation": "ML evaluation"
        })


# ------------------------------------------
# CNN Model
# ------------------------------------------

if cnn_results:

    models.append({
        "name": "CNN",

        "accuracy": find_metric(
            cnn_results,
            {
                "accuracy",
                "validationaccuracy",
                "valaccuracy"
            }
        ),

        "precision": find_metric(
            cnn_results,
            {
                "precision",
                "weightedprecision"
            }
        ),

        "recall": find_metric(
            cnn_results,
            {
                "recall",
                "weightedrecall"
            }
        ),

        "f1_score": find_metric(
            cnn_results,
            {
                "f1score",
                "weightedf1",
                "f1"
            }
        ),

        "evaluation": "Validation"
    })


# ------------------------------------------
# YOLOv8 Classification Model
# ------------------------------------------

if yolo_results:

    models.append({
        "name": "YOLOv8 Classification",

        "accuracy": find_metric(
            yolo_results,
            {
                "testtop1accuracy",
                "top1accuracy",
                "accuracytop1",
                "metricsaccuracytop1",
                "accuracy"
            }
        ),

        "precision": find_metric(
            yolo_results,
            {
                "precision",
                "weightedprecision"
            }
        ),

        "recall": find_metric(
            yolo_results,
            {
                "recall",
                "weightedrecall"
            }
        ),

        "f1_score": find_metric(
            yolo_results,
            {
                "f1score",
                "weightedf1",
                "f1"
            }
        ),

        "evaluation": "Test"
    })


# ==========================================
# 5. DISPLAY COMPARISON TABLE
# ==========================================

print("\n" + "=" * 90)
print("             WASTE CLASSIFICATION MODEL COMPARISON")
print("=" * 90)

print(
    f"{'Model':<27}"
    f"{'Accuracy':>12}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1 Score':>12}"
)

print("-" * 90)


for model in models:

    accuracy = (
        f"{model['accuracy']:.2f}%"
        if model["accuracy"] is not None else "N/A"
    )

    precision = (
        f"{model['precision']:.2f}%"
        if model["precision"] is not None else "N/A"
    )

    recall = (
        f"{model['recall']:.2f}%"
        if model["recall"] is not None else "N/A"
    )

    f1_score = (
        f"{model['f1_score']:.2f}%"
        if model["f1_score"] is not None else "N/A"
    )

    print(
        f"{model['name']:<27}"
        f"{accuracy:>12}"
        f"{precision:>12}"
        f"{recall:>12}"
        f"{f1_score:>12}"
    )


print("-" * 90)

print("\nEVALUATION TYPES")

for model in models:

    print(
        f"{model['name']}: {model['evaluation']}"
    )


# ==========================================
# 6. GENERATE COMPARISON CHARTS
# ==========================================

metrics = {
    "accuracy": "Accuracy Comparison",
    "precision": "Precision Comparison",
    "recall": "Recall Comparison",
    "f1_score": "F1-Score Comparison"
}


os.makedirs(RESULTS_DIR, exist_ok=True)


for metric, title in metrics.items():

    available_models = [
        model for model in models
        if model[metric] is not None
    ]

    if not available_models:

        print(f"\nNo data available for {title}")
        continue

    names = [
        model["name"]
        for model in available_models
    ]

    values = [
        model[metric]
        for model in available_models
    ]

    plt.figure(figsize=(12, 6))

    bars = plt.bar(names, values)

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Classification Models")
    plt.ylabel("Percentage (%)")

    plt.ylim(0, 105)

    plt.xticks(
        rotation=20,
        ha="right"
    )

    # Display percentages above bars
    for bar, value in zip(bars, values):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.2f}%",
            ha="center",
            fontsize=9
        )

    plt.tight_layout()

    filename = f"{metric}_comparison.png"

    save_path = os.path.join(
        RESULTS_DIR,
        filename
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(f"Chart saved: {save_path}")


# ==========================================
# 7. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 90)
print("MODEL COMPARISON COMPLETED SUCCESSFULLY!")
print("=" * 90)

print("\nGenerated charts:")

for metric in metrics:

    print(
        f"- {metric}_comparison.png"
    )

print("\nAll available model results have been processed.")
