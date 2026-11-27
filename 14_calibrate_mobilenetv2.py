"""Choose a manual-review threshold using validation images only."""
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
BASE=Path(__file__).resolve().parent; DATA=BASE/"shared_dataset"
MODEL=BASE/"models"/"mobilenetv2_waste_model.keras"; RESULTS=BASE/"results"
CLASSES=["cardboard","glass","metal","paper","plastic","trash"]

def main():
    if not MODEL.is_file(): raise FileNotFoundError("Train MobileNetV2 first.")
    ds=tf.keras.utils.image_dataset_from_directory(DATA/"val",class_names=CLASSES,
      image_size=(224,224),batch_size=32,shuffle=False)
    y=np.concatenate([labels.numpy() for _,labels in ds])
    probs=tf.keras.models.load_model(MODEL).predict(ds,verbose=1)
    pred=probs.argmax(1); confidence=probs.max(1); chosen=None
    for threshold in np.unique(np.round(confidence,4)):
        keep=confidence>=threshold
        if keep.sum()>=max(5,len(CLASSES)):
            precision=float(np.mean(pred[keep]==y[keep]))
            if precision>=.90:
                chosen=(float(threshold),precision,float(keep.mean())); break
    if chosen:
        threshold,precision,coverage=chosen; met=True
    else:
        threshold=float(np.quantile(confidence,.90)); keep=confidence>=threshold
        precision=float(np.mean(pred[keep]==y[keep])) if keep.any() else 0.0
        coverage=float(keep.mean()); met=False
    payload={"threshold":threshold,"validation_precision":precision,
      "validation_coverage":coverage,"target_precision":.90,"target_met":met,
      "selection_split":"validation"}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS/"mobilenetv2_confidence_calibration.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(payload)

if __name__=="__main__": main()
