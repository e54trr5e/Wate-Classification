"""Train YOLOv8 image classification on the shared train/validation/test split."""
from pathlib import Path
from ultralytics import YOLO
BASE=Path(__file__).resolve().parent
DATA=BASE/"shared_dataset"
MODEL_NAME="yolov8n-cls.pt"
EPOCHS=30; IMAGE_SIZE=224; BATCH_SIZE=16

def main():
    if not (DATA/"manifest.csv").is_file():
        raise FileNotFoundError("Run 00_prepare_shared_dataset.py first.")
    for split in ("train","val","test"):
        for name in ("cardboard","glass","metal","paper","plastic","trash"):
            if not (DATA/split/name).is_dir(): raise FileNotFoundError(DATA/split/name)
    model=YOLO(MODEL_NAME)
    model.train(data=str(DATA),epochs=EPOCHS,imgsz=IMAGE_SIZE,batch=BATCH_SIZE,
      workers=0,project=str(BASE/"runs"/"yolo_classification"),
      name="waste_yolov8n_cls",exist_ok=True,patience=7,plots=True,seed=42)
    best=BASE/"runs"/"yolo_classification"/"waste_yolov8n_cls"/"weights"/"best.pt"
    if not best.is_file(): raise FileNotFoundError(f"Best weights not found: {best}")
    print("Training complete. Run 10_evaluate_yolo.py for held-out test metrics.")
    print("Best model:",best)

if __name__=="__main__": main()
