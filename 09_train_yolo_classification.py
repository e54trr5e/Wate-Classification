
import os
import json
from ultralytics import YOLO

# -------------------------------
# CONFIGURATION
# -------------------------------

DATASET_DIR = "./yolo_classification"
MODEL_NAME = "yolov8n-cls.pt"

EPOCHS = 30
IMAGE_SIZE = 224
BATCH_SIZE = 16


def main():

    print("=" * 55)
    print("YOLOv8 WASTE CLASSIFICATION TRAINING")
    print("=" * 55)

    # Check dataset folders
    for split in ["train", "val", "test"]:
        split_path = os.path.join(DATASET_DIR, split)

        if not os.path.isdir(split_path):
            raise FileNotFoundError(
                f"Missing dataset folder: {split_path}"
            )

    # Load pretrained YOLOv8 classification model
    print("\nLoading pretrained YOLOv8 model...")

    model = YOLO(MODEL_NAME)

    # Train model
    print("\nStarting YOLOv8 training...\n")

    model.train(
        data=DATASET_DIR,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=0,
        project="./runs/yolo_classification",
        name="waste_yolov8n_cls",
        exist_ok=True,
        patience=7,
        plots=True,
        seed=42
    )

    # Load best trained model
    best_model_path = (
    "./runs/classify/runs/yolo_classification/"
    "waste_yolov8n_cls/weights/best.pt"
)

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            "Best YOLO model weights were not found."
        )

    print("\nLoading best trained model...")

    best_model = YOLO(best_model_path)

    # Evaluate on separate test dataset
    print("\nEvaluating on test dataset...\n")

    metrics = best_model.val(
        data=DATASET_DIR,
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=0
    )

    top1_accuracy = float(metrics.top1)
    top5_accuracy = float(metrics.top5)

    # Display results
    print("\n" + "=" * 55)
    print("YOLOv8 CLASSIFICATION RESULTS")
    print("=" * 55)

    print(f"Test Top-1 Accuracy: {top1_accuracy * 100:.2f}%")
    print(f"Test Top-5 Accuracy: {top5_accuracy * 100:.2f}%")

    # Save results
    os.makedirs("./results", exist_ok=True)

    results = {
        "model": "YOLOv8n Classification",
        "test_top1_accuracy": top1_accuracy,
        "test_top5_accuracy": top5_accuracy,
        "epochs": EPOCHS,
        "image_size": IMAGE_SIZE
    }

    with open(
        "./results/yolo_classification_results.json",
        "w"
    ) as file:
        json.dump(results, file, indent=4)

    print("\nResults saved successfully!")
    print("Location: ./results/yolo_classification_results.json")


if __name__ == "__main__":
    main()
