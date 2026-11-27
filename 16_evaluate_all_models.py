"""Compare models only when every saved result has the identical shared test split."""
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent; RESULTS=BASE/"results"
SPLIT="shared_dataset_seed_42_70_15_15"
FILES=["ml_model_results.json","cnn_results.json","mobilenetv2_results.json","yolo_classification_results.json"]

def main():
    if not (BASE/"shared_dataset"/"manifest.csv").is_file(): raise FileNotFoundError("Run shared split preparation first.")
    rows=[]
    for filename in FILES:
        path=RESULTS/filename
        if not path.is_file(): print("Not evaluated yet:",filename); continue
        data=json.loads(path.read_text(encoding="utf-8"))
        if data.get("evaluation")!="test" or data.get("split_id")!=SPLIT:
            raise ValueError(f"{filename} is not from the common held-out test set.")
        if filename=="ml_model_results.json":
            rows.extend((name,item) for name,item in data["models"].items())
        else: rows.append((data.get("model",filename),data))
    if not rows: raise RuntimeError("No common-split results are available.")
    print("\nCOMMON TEST SET MODEL COMPARISON")
    print(f"{'Model':30} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>13}")
    for name,item in rows:
        print(f"{name:30} {item.get('accuracy',0)*100:9.2f}% {item.get('macro_f1',0)*100:9.2f}% {item.get('weighted_f1',0)*100:12.2f}%")
    payload={"evaluation":"test","split_id":SPLIT,"models":[{"name":name,**item} for name,item in rows]}
    (RESULTS/"final_model_comparison.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print("\nSaved results/final_model_comparison.json")

if __name__=="__main__": main()
