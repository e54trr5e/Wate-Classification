"""Create one deterministic, stratified train/validation/test dataset for every model."""
import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path

SOURCE = Path("dataset")
OUTPUT = Path("shared_dataset")
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
RATIOS = (0.70, 0.15, 0.15)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Replace the generated shared_dataset directory.")
    args = parser.parse_args()

    if not SOURCE.is_dir():
        raise FileNotFoundError("Expected source dataset/<class_name>/ folders.")

    if OUTPUT.exists():
        if not args.force:
            raise FileExistsError(
                "shared_dataset already exists. Use --force only if you intend "
                "to regenerate the split and overwrite the previous manifest."
            )
        shutil.rmtree(OUTPUT)

    rng = random.Random(SEED)
    manifest = []
    seen_hashes = {}

    for class_name in CLASSES:
        folder = SOURCE / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class directory: {folder}")

        images = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )
        if len(images) < 3:
            raise ValueError(f"{class_name} needs at least 3 images; found {len(images)}.")

        # Exact duplicates across the source dataset can leak across splits.
        unique_images = []
        for image in images:
            digest = sha256(image)
            if digest in seen_hashes:
                print(f"Duplicate skipped: {image} == {seen_hashes[digest]}")
                continue
            seen_hashes[digest] = image
            unique_images.append(image)

        if len(unique_images) < 3:
            raise ValueError(f"{class_name} has fewer than 3 unique images.")

        rng.shuffle(unique_images)
        n = len(unique_images)
        n_train = int(n * RATIOS[0])
        n_val = int(n * RATIOS[1])
        # Guarantee all three partitions receive at least one image.
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
        partitions = {
            "train": unique_images[:n_train],
            "val": unique_images[n_train:n_train + n_val],
            "test": unique_images[n_train + n_val:],
        }

        for split, paths in partitions.items():
            target_dir = OUTPUT / split / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for index, source_path in enumerate(paths):
                # Prefix index to avoid same-name collisions after recursive scan.
                target_name = f"{index:05d}_{source_path.name}"
                target_path = target_dir / target_name
                shutil.copy2(source_path, target_path)
                manifest.append({
                    "split": split,
                    "class_name": class_name,
                    "label": CLASSES.index(class_name),
                    "source_path": str(source_path.resolve()),
                    "image_path": str(target_path.as_posix()),
                    "sha256": sha256(source_path),
                })

        print(
            f"{class_name:10} train={len(partitions['train']):4} "
            f"val={len(partitions['val']):4} test={len(partitions['test']):4}"
        )

    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    print(f"\nCreated {OUTPUT} with {len(manifest)} unique images.")
    print(f"Seed: {SEED}; split ratios: train/val/test = {RATIOS}")
    print("All model families must use these folders; do not regenerate between runs.")


if __name__ == "__main__":
    main()
