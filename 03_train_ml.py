"""Train traditional classifiers on the shared split; choose the best by validation macro-F1."""
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, precision_score, recall_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

BASE=Path(__file__).resolve().parent
RESULTS=BASE/"results"; MODELS=BASE/"models"
CLASSES=["cardboard","glass","metal","paper","plastic","trash"]
SEED=42

def score(y,p):
    return {"accuracy":float(accuracy_score(y,p)),
      "macro_precision":float(precision_score(y,p,average="macro",zero_division=0)),
      "macro_recall":float(recall_score(y,p,average="macro",zero_division=0)),
      "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
      "weighted_f1":float(f1_score(y,p,average="weighted",zero_division=0)),
      "classification_report":classification_report(y,p,labels=list(range(6)),target_names=CLASSES,output_dict=True,zero_division=0),
      "confusion_matrix":confusion_matrix(y,p,labels=list(range(6))).tolist()}

def calibrate(y,probs,target=0.90):
    conf=probs.max(axis=1); pred=probs.argmax(axis=1)
    for threshold in np.unique(np.round(conf,4)):
        keep=conf>=threshold
        if keep.sum()>=max(5,len(CLASSES)):
            precision=float(np.mean(pred[keep]==y[keep]))
            if precision>=target:
                return {"threshold":float(threshold),"validation_precision":precision,
                    "validation_coverage":float(keep.mean()),"target_precision":target,"target_met":True}
    threshold=float(np.quantile(conf,0.90)); keep=conf>=threshold
    precision=float(np.mean(pred[keep]==y[keep])) if keep.any() else 0.0
    return {"threshold":threshold,"validation_precision":precision,
        "validation_coverage":float(keep.mean()),"target_precision":target,"target_met":False}

def main():
    feature_file=RESULTS/"shared_features.pkl"
    if not feature_file.is_file(): raise FileNotFoundError("Run 02_extract_shared_features.py first.")
    with feature_file.open("rb") as f: data=pickle.load(f)
    X,y,s=data["X"],data["y"],data["split"]
    tr,va,te=s=="train",s=="val",s=="test"
    if not all(mask.any() for mask in (tr,va,te)): raise ValueError("Shared split is incomplete.")
    models={
      "Random Forest":RandomForestClassifier(n_estimators=400,class_weight="balanced_subsample",random_state=SEED,n_jobs=-1),
      "KNN":Pipeline([("scale",StandardScaler()),("pca",PCA(n_components=0.95,svd_solver="full")),("model",KNeighborsClassifier(n_neighbors=5,weights="distance"))]),
      "Logistic Regression":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=2000,class_weight="balanced",random_state=SEED))]),
      "SVM":Pipeline([("scale",StandardScaler()),("pca",PCA(n_components=0.95,svd_solver="full")),("model",SVC(kernel="rbf",probability=True,class_weight="balanced",random_state=SEED))])}
    MODELS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    results={}; val_scores={}
    for name,model in models.items():
        print("Training",name); model.fit(X[tr],y[tr])
        val_scores[name]=float(f1_score(y[va],model.predict(X[va]),average="macro",zero_division=0))
        item=score(y[te],model.predict(X[te]))
        item.update({"model":name,"evaluation":"test","test_count":int(te.sum()),
          "split_id":"shared_dataset_seed_42_70_15_15","validation_macro_f1_for_selection":val_scores[name]})
        results[name]=item
        fig,ax=plt.subplots(figsize=(8,6))
        ConfusionMatrixDisplay(np.asarray(item["confusion_matrix"]),display_labels=CLASSES).plot(ax=ax,xticks_rotation=35)
        ax.set_title(name+" — Shared Test Split"); fig.tight_layout()
        fig.savefig(RESULTS/(name.lower().replace(" ","_")+"_confusion_matrix.png"),dpi=220); plt.close(fig)
        print("Validation macro-F1",val_scores[name],"Test macro-F1",item["macro_f1"])
    best_name=max(val_scores,key=val_scores.get); best=models[best_name]
    calibration=calibrate(y[va],best.predict_proba(X[va]))
    bundle={"model":best,"classes":CLASSES,"feature_count":int(X.shape[1]),"model_name":best_name,
      "confidence_calibration":calibration,"selection_metric":"validation_macro_f1",
      "evaluation_split":"shared_dataset/test"}
    with (MODELS/"best_ml_model.pkl").open("wb") as f: pickle.dump(bundle,f)
    payload={"evaluation":"test","split_id":"shared_dataset_seed_42_70_15_15",
      "test_count":int(te.sum()),"classes":CLASSES,"models":results,
      "selected_model":best_name,"selection_metric":"validation_macro_f1",
      "confidence_calibration":calibration}
    (RESULTS/"ml_model_results.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    (RESULTS/"ml_confidence_calibration.json").write_text(json.dumps(calibration,indent=2),encoding="utf-8")
    print("Best model:",best_name,"Validation-calibrated review threshold:",calibration)

if __name__=="__main__": main()
