"""Predict an image with the selected traditional ML model and calibrated review threshold."""
import argparse
import importlib.util
import pickle
from pathlib import Path
import cv2
import numpy as np
BASE=Path(__file__).resolve().parent
MODEL=BASE/"models"/"best_ml_model.pkl"
FEATURE_SCRIPT=BASE/"02_extract_features.py"

def extractor():
    spec=importlib.util.spec_from_file_location("waste_features",FEATURE_SCRIPT)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.extract_features

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("image"); args=parser.parse_args()
    image_path=Path(args.image).expanduser()
    if not image_path.is_file(): raise FileNotFoundError(image_path)
    if not MODEL.is_file(): raise FileNotFoundError("Train ML models first.")
    with MODEL.open("rb") as f: bundle=pickle.load(f)
    model=bundle["model"]; classes=bundle["classes"]
    image=cv2.imread(str(image_path))
    if image is None: raise ValueError("Could not read image.")
    features=extractor()(image).reshape(1,-1)
    if features.shape[1]!=bundle["feature_count"]: raise ValueError("Feature count mismatch.")
    probabilities=model.predict_proba(features)[0]
    mapping={int(c):float(v) for c,v in zip(model.classes_,probabilities)}
    probs=np.asarray([mapping.get(i,0.0) for i in range(len(classes))])
    index=int(probs.argmax()); confidence=float(probs[index])
    threshold=float(bundle.get("confidence_calibration",{}).get("threshold",.5))
    print("Selected model:",bundle.get("model_name","Traditional ML"))
    print("Prediction:",classes[index].upper(),f"({confidence*100:.2f}%)")
    print("Validation-calibrated manual-review threshold:",f"{threshold*100:.2f}%")
    if confidence<threshold: print("WARNING: Low confidence. Manual verification recommended.")
    print("\nTop predictions:")
    for i in probs.argsort()[::-1][:3]: print(f"{classes[int(i)]}: {probs[i]*100:.2f}%")

if __name__=="__main__": main()
