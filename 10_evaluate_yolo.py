import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    BASE_DIR / "runs" / "yolo_classification"
    / "waste_yolov8n_cls" / "weights" / "best.pt"
)
DATASET_DIR = BASE_DIR / "yolo_classification"
TEST_DIR = DATASET_DIR / "test"
RESULTS_DIR = BASE_DIR / "results"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_test_images():
    image_paths = []
    true_labels = []

    for class_index, class_name in enumerate(CLASSES):
        class_dir = TEST_DIR / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing test class folder: {class_dir}")

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(image_path)
                true_labels.append(class_index)

    if not image_paths:
        raise ValueError(f"No test images found under {TEST_DIR}")

    return image_paths, np.asarray(true_labels, dtype=np.int64)


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained YOLO model not found: {MODEL_PATH}. "
            "Run 09_train_yolo_classification.py first."
        )

    image_paths, y_true = collect_test_images()
    model = YOLO(str(MODEL_PATH))

    print(f"Evaluating {len(image_paths)} test images...")
    predictions = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=224,
        batch=16,
        workers=0,
        verbose=False,
    )

    y_pred = []
    for result in predictions:
        if result.probs is None:
            raise RuntimeError(f"Classification probabilities missing for {result.path}")
        y_pred.append(int(result.probs.top1))

    y_pred = np.asarray(y_pred, dtype=np.int64)

    metrics = {
        "model": "YOLOv8n Classification",
        "evaluation": "test",
        "test_image_count": int(len(y_true)),
        "classes": CLASSES,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, labels=list(range(len(CLASSES))),
            target_names=CLASSES, output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(CLASSES)))
        ).tolist(),
    }

    print("\nYOLOv8 TEST CLASSIFICATION REPORT")
    print(classification_report(
        y_true, y_pred, labels=list(range(len(CLASSES))),
        target_names=CLASSES, zero_division=0,
    ))
    print("Macro F1:", f"{metrics['macro_f1'] * 100:.2f}%")
    print("Weighted F1:", f"{metrics['weighted_f1'] * 100:.2f}%")

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "yolo_classification_results.json"
    output_path.write_text(json.dumps(metrics, indent=4), encoding="utf-8")
    print("\nSaved detailed evaluation:", output_path)


if __name__ == "__main__":
    main()
