
import random
import shutil
from pathlib import Path

SOURCE_DIR = Path("./dataset")
OUTPUT_DIR = Path("./yolo_classification")

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
random.seed(42)


def main():
    if OUTPUT_DIR.exists():
        print("YOLO dataset folder already exists.")
        print("Stopping to avoid copying duplicate images.")
        return

    for class_name in CLASSES:
        source = SOURCE_DIR / class_name

        if not source.is_dir():
            raise FileNotFoundError(f"Missing class folder: {source}")

        images = [
            p for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if len(images) < 3:
            raise ValueError(
                f"Not enough images in {class_name}: {len(images)}"
            )

        random.shuffle(images)

        train_end = int(len(images) * TRAIN_RATIO)
        val_end = train_end + int(len(images) * VAL_RATIO)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split_name, split_images in splits.items():
            destination = OUTPUT_DIR / split_name / class_name
            destination.mkdir(parents=True, exist_ok=True)

            for image_path in split_images:
                shutil.copy2(image_path, destination / image_path.name)

        print(
            f"{class_name}: "
            f"Train={len(splits['train'])}, "
            f"Val={len(splits['val'])}, "
            f"Test={len(splits['test'])}"
        )

    print("\nYOLO classification dataset prepared successfully!")


if __name__ == "__main__":
    main()
