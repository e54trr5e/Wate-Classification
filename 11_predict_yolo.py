
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = (
    "./runs/classify/runs/yolo_classification/"
    "waste_yolov8n_cls/weights/best.pt"
)

def main():
    print("=" * 45)
    print("YOLOv8 WASTE CLASSIFIER")
    print("=" * 45)

    image_path = input("\nEnter the image path: ").strip().strip('"')

    if not Path(image_path).is_file():
        print("Error: Image file not found!")
        return

    print("\nLoading trained YOLOv8 model...")

    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=image_path,
        imgsz=224,
        verbose=False
    )

    result = results[0]
    probabilities = result.probs.data.cpu().numpy()

    predicted_id = int(result.probs.top1)
    predicted_class = result.names[predicted_id]
    confidence = float(result.probs.top1conf)

    print("\n" + "=" * 45)
    print("PREDICTION RESULT")
    print("=" * 45)

    print(f"Predicted Category: {predicted_class.upper()}")
    print(f"Confidence Score: {confidence * 100:.2f}%")

    print("\nTOP 3 PREDICTIONS")

    top_indices = probabilities.argsort()[::-1][:3]

    for index in top_indices:
        class_name = result.names[int(index)]
        score = float(probabilities[index]) * 100
        print(f"{class_name}: {score:.2f}%")

    print("\nPrediction completed successfully!")


if __name__ == "__main__":
    main()
