"""Extract handcrafted features from the common shared_dataset split."""
import importlib.util
import pickle
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
DATA = BASE / "shared_dataset"
RESULTS = BASE / "results"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_extractor():
    spec = importlib.util.spec_from_file_location(
        "waste_feature_module", BASE / "02_extract_features.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_features


def main():
    if not (DATA / "manifest.csv").is_file():
        raise FileNotFoundError("Run 00_prepare_shared_dataset.py first.")
    extract = load_extractor()
    features, labels, splits, paths = [], [], [], []

    for split in ("train", "val", "test"):
        for label, class_name in enumerate(CLASSES):
            folder = DATA / split / class_name
            images = sorted(
                p for p in folder.rglob("*")
                if p.is_file() and p.suffix.lower() in EXTENSIONS
            )
            for image_path in images:
                import cv2
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Unreadable image in fixed split: {image_path}")
                features.append(extract(image))
                labels.append(label)
                splits.append(split)
                paths.append(str(image_path))

    if not features:
        raise ValueError("No images found in shared_dataset.")
    RESULTS.mkdir(exist_ok=True)
    payload = {
        "X": np.asarray(features, dtype=np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "split": np.asarray(splits),
        "paths": np.asarray(paths),
        "classes": CLASSES,
    }
    with (RESULTS / "shared_features.pkl").open("wb") as file:
        pickle.dump(payload, file)
    print("Saved", len(labels), "feature vectors to results/shared_features.pkl")
    for split in ("train", "val", "test"):
        print(split, int(np.sum(payload["split"] == split)))


if __name__ == "__main__":
    main()
