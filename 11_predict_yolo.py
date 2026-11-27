"""Predict one image using the trained YOLOv8 classification model."""
from pathlib import Path
from ultralytics import YOLO
BASE=Path(__file__).resolve().parent
MODEL=BASE/"runs"/"yolo_classification"/"waste_yolov8n_cls"/"weights"/"best.pt"

def main():
    image=Path(input("Enter image path: ").strip().strip('"')).expanduser()
    if not image.is_file(): raise FileNotFoundError(image)
    if not MODEL.is_file(): raise FileNotFoundError("Train YOLO first: "+str(MODEL))
    result=YOLO(str(MODEL)).predict(source=str(image),imgsz=224,verbose=False)[0]
    if result.probs is None: raise RuntimeError("No classification probabilities returned.")
    probs=result.probs.data.cpu().numpy()
    print("\nPredicted:",result.names[int(result.probs.top1)].upper())
    print("Confidence:",f"{float(result.probs.top1conf)*100:.2f}%")
    print("\nTop predictions:")
    for index in probs.argsort()[::-1][:3]:
        print(f"{result.names[int(index)]}: {float(probs[index])*100:.2f}%")

if __name__=="__main__": main()
