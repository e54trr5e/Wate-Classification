"""Validate the generated common split manifest and guard against data leakage."""
import csv
import hashlib
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent
MANIFEST=ROOT/"shared_dataset"/"manifest.csv"
EXPECTED={"train","val","test"}
CLASSES={"cardboard","glass","metal","paper","plastic","trash"}

def main():
    if not MANIFEST.is_file(): raise FileNotFoundError("Run 00_prepare_shared_dataset.py first.")
    with MANIFEST.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    if not rows: raise ValueError("Manifest is empty.")
    hashes={}; paths=set(); counts=Counter()
    for row in rows:
        split=row["split"]; label=row["class_name"]; path=Path(row["image_path"])
        if split not in EXPECTED: raise ValueError(f"Unknown split: {split}")
        if label not in CLASSES: raise ValueError(f"Unknown class: {label}")
        if not path.is_file(): raise FileNotFoundError(path)
        key=row["sha256"]
        if key in hashes and hashes[key]!=split:
            raise ValueError(f"Data leakage: duplicate image appears in {hashes[key]} and {split}")
        hashes[key]=split
        if row["image_path"] in paths: raise ValueError(f"Repeated path: {path}")
        paths.add(row["image_path"]); counts[(split,label)]+=1
    for label in sorted(CLASSES):
        values=[counts[(s,label)] for s in ("train","val","test")]
        if any(v==0 for v in values): raise ValueError(f"{label} has an empty split: {values}")
        print(f"{label:10} train={values[0]:4} val={values[1]:4} test={values[2]:4}")
    print(f"\nSplit valid: {len(rows)} images, {len(hashes)} unique SHA-256 hashes.")
    print("Use this same shared_dataset for every model run.")

if __name__=="__main__": main()
