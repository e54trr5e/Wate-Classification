
import json
import os
from ultralytics import YOLO

MODEL_PATH = (
    "./runs/classify/runs/yolo_classification/"
    "waste_yolov8n_cls/weights/best.pt"
)

DATASET_DIR = "./yolo_classification"

print("Loading trained YOLOv8 model...")

model = YOLO(MODEL_PATH)

print("\nEvaluating on test dataset...")

metrics = model.val(
    data=DATASET_DIR,
    split="test",
    imgsz=224,
    batch=16,
    workers=0
)

top1 = float(metrics.top1)
top5 = float(metrics.top5)

print("\n" + "=" * 45)
print("YOLOv8 TEST RESULTS")
print("=" * 45)

print(f"Top-1 Accuracy: {top1 * 100:.2f}%")
print(f"Top-5 Accuracy: {top5 * 100:.2f}%")

os.makedirs("./results", exist_ok=True)

with open("./results/yolo_classification_results.json", "w") as f:
    json.dump({
        "model": "YOLOv8n Classification",
        "test_top1_accuracy": top1,
        "test_top5_accuracy": top5
    }, f, indent=4)

print("\nResults saved successfully!")
