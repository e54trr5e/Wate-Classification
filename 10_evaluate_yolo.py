"""Evaluate YOLOv8 classification on the common held-out test set."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,
    f1_score,precision_score,recall_score,ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
from ultralytics import YOLO
BASE=Path(__file__).resolve().parent
MODEL=BASE/"runs"/"yolo_classification"/"waste_yolov8n_cls"/"weights"/"best.pt"
TEST=BASE/"shared_dataset"/"test"; RESULTS=BASE/"results"
CLASSES=["cardboard","glass","metal","paper","plastic","trash"]
EXT={".jpg",".jpeg",".png",".bmp",".webp"}

def main():
    if not (BASE/"shared_dataset"/"manifest.csv").is_file(): raise FileNotFoundError("Prepare shared split first.")
    if not MODEL.is_file(): raise FileNotFoundError("Train YOLO first: "+str(MODEL))
    paths=[]; truth=[]
    for label,name in enumerate(CLASSES):
        folder=TEST/name
        if not folder.is_dir(): raise FileNotFoundError(folder)
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() in EXT: paths.append(p); truth.append(label)
    if not paths: raise ValueError("No test images found.")
    model=YOLO(str(MODEL)); results=model.predict(source=[str(p) for p in paths],
      imgsz=224,batch=16,workers=0,verbose=False)
    pred=[]
    for result in results:
        if result.probs is None: raise RuntimeError("YOLO returned no classification probabilities.")
        pred.append(int(result.probs.top1))
    y=np.asarray(truth); p=np.asarray(pred); cm=confusion_matrix(y,p,labels=list(range(6)))
    report={"model":"YOLOv8n Classification","evaluation":"test",
      "split_id":"shared_dataset_seed_42_70_15_15","test_count":len(y),"classes":CLASSES,
      "accuracy":float(accuracy_score(y,p)),
      "macro_precision":float(precision_score(y,p,average="macro",zero_division=0)),
      "macro_recall":float(recall_score(y,p,average="macro",zero_division=0)),
      "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
      "weighted_f1":float(f1_score(y,p,average="weighted",zero_division=0)),
      "classification_report":classification_report(y,p,labels=list(range(6)),target_names=CLASSES,output_dict=True,zero_division=0),
      "confusion_matrix":cm.tolist()}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS/"yolo_classification_results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    ConfusionMatrixDisplay(cm,display_labels=CLASSES).plot(xticks_rotation=35)
    plt.tight_layout(); plt.savefig(RESULTS/"yolo_confusion_matrix.png",dpi=220); plt.close()
    print(classification_report(y,p,labels=list(range(6)),target_names=CLASSES,zero_division=0))
    print("YOLO test macro-F1:",report["macro_f1"])

if __name__=="__main__": main()
