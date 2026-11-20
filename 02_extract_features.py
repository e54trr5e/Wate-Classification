
import os
import cv2
import pickle
import numpy as np

from pathlib import Path


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_DIR = "./dataset"
RESULTS_DIR = "./results"

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

IMG_SIZE = (128, 128)

SUPPORTED_FORMATS = (
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================================
# 2. HSV COLOR HISTOGRAM
# ==========================================

def extract_hsv_features(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 180, 0, 256, 0, 256]
    )

    hist = hist.flatten().astype(np.float32)

    hist /= (hist.sum() + 1e-7)

    return hist


# ==========================================
# 3. RGB COLOR HISTOGRAM
# ==========================================

def extract_rgb_features(image):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    features = []

    for channel in range(3):

        hist = cv2.calcHist(
            [rgb],
            [channel],
            None,
            [8],
            [0, 256]
        )

        hist = hist.flatten()

        hist = hist / (hist.sum() + 1e-7)

        features.extend(hist)

    return np.array(features, dtype=np.float32)


# ==========================================
# 4. COLOR STATISTICS
# ==========================================

def extract_color_statistics(image):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    rgb = rgb.astype(np.float32) / 255.0

    features = []

    for channel in range(3):

        channel_data = rgb[:, :, channel]

        mean = np.mean(channel_data)
        std = np.std(channel_data)

        features.append(mean)
        features.append(std)

    return np.array(features, dtype=np.float32)


# ==========================================
# 5. HOG SHAPE AND EDGE FEATURES
# ==========================================

def extract_hog_features(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Calculate horizontal and vertical gradients
    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude, angle = cv2.cartToPolar(
        gx,
        gy,
        angleInDegrees=True
    )

    # Unsigned gradient orientations: 0 to 180
    angle = angle % 180

    cell_size = 16
    bins = 9

    height, width = gray.shape

    hog_features = []

    # Divide image into 8 x 8 cells
    for y in range(0, height, cell_size):

        for x in range(0, width, cell_size):

            cell_angle = angle[
                y:y + cell_size,
                x:x + cell_size
            ]

            cell_magnitude = magnitude[
                y:y + cell_size,
                x:x + cell_size
            ]

            histogram, _ = np.histogram(
                cell_angle,
                bins=bins,
                range=(0, 180),
                weights=cell_magnitude
            )

            # Normalize each cell histogram
            histogram = histogram.astype(np.float32)

            histogram /= (
                np.linalg.norm(histogram) + 1e-7
            )

            hog_features.extend(histogram)

    return np.array(
        hog_features,
        dtype=np.float32
    )


# ==========================================
# 6. LBP TEXTURE FEATURES
# ==========================================

def extract_lbp_features(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = gray.astype(np.int16)

    center = gray[1:-1, 1:-1]

    # Eight neighboring pixels
    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1)
    ]

    lbp = np.zeros(
        center.shape,
        dtype=np.uint8
    )

    for bit, (dy, dx) in enumerate(neighbors):

        neighbor = gray[
            1 + dy:gray.shape[0] - 1 + dy,
            1 + dx:gray.shape[1] - 1 + dx
        ]

        lbp |= (
            (neighbor >= center).astype(np.uint8)
            << bit
        )

    # Generate 256-bin LBP histogram
    hist, _ = np.histogram(
        lbp,
        bins=256,
        range=(0, 256)
    )

    hist = hist.astype(np.float32)

    hist /= (hist.sum() + 1e-7)

    return hist


# ==========================================
# 7. EDGE DENSITY
# ==========================================

def extract_edge_features(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_density = np.mean(
        edges > 0
    )

    return np.array(
        [edge_density],
        dtype=np.float32
    )


# ==========================================
# 8. COMBINE ALL FEATURES
# ==========================================

def extract_features(image):

    if image is None:
        raise ValueError("Invalid input image.")

    image = cv2.resize(
        image,
        IMG_SIZE
    )

    hsv_features = extract_hsv_features(image)

    rgb_features = extract_rgb_features(image)

    color_features = extract_color_statistics(image)

    hog_features = extract_hog_features(image)

    lbp_features = extract_lbp_features(image)

    edge_features = extract_edge_features(image)

    # Combine every feature into one vector
    combined_features = np.concatenate([
        hsv_features,
        rgb_features,
        color_features,
        hog_features,
        lbp_features,
        edge_features
    ])

    return combined_features.astype(np.float32)


# ==========================================
# 9. GENERATE FEATURE NAMES
# ==========================================

def generate_feature_names():

    names = []

    # HSV: 8 x 8 x 8 = 512
    for h in range(8):

        for s in range(8):

            for v in range(8):

                names.append(
                    f"HSV_H{h}_S{s}_V{v}"
                )

    # RGB: 3 x 8 = 24
    for channel in ["Red", "Green", "Blue"]:

        for bin_index in range(8):

            names.append(
                f"{channel}_Histogram_{bin_index}"
            )

    # RGB statistics: 6
    for channel in ["Red", "Green", "Blue"]:

        names.append(f"{channel}_Mean")
        names.append(f"{channel}_Std")

    # HOG: 8 x 8 x 9 = 576
    for row in range(8):

        for col in range(8):

            for orientation in range(9):

                names.append(
                    f"HOG_Cell_{row}_{col}_Bin_{orientation}"
                )

    # LBP: 256
    for i in range(256):

        names.append(
            f"LBP_Histogram_{i}"
        )

    # Edge density: 1
    names.append("Edge_Density")

    return names


# ==========================================
# 10. PROCESS COMPLETE DATASET
# ==========================================

def process_dataset():

    features = []
    labels = []

    print("\n" + "=" * 60)
    print("SMART WASTE - ADVANCED FEATURE EXTRACTION")
    print("=" * 60)

    for label_index, class_name in enumerate(CLASSES):

        class_path = Path(DATASET_DIR) / class_name

        if not class_path.exists():

            print(
                f"\nWARNING: Missing folder: {class_name}"
            )

            continue

        print(
            f"\nProcessing category: {class_name.upper()}"
        )

        image_count = 0
        skipped_count = 0

        image_paths = sorted(
            p for p in class_path.rglob("*")
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_FORMATS
        )

        for image_path in image_paths:

            image = cv2.imread(str(image_path))

            if image is None:

                skipped_count += 1
                continue

            try:

                feature_vector = extract_features(image)

                features.append(feature_vector)

                labels.append(label_index)

                image_count += 1

            except Exception as error:

                skipped_count += 1

                print(
                    "Skipped:",
                    image_path.name,
                    "|",
                    error
                )

        print("Successfully processed:", image_count)
        print("Skipped images:", skipped_count)

    if len(features) == 0:

        raise ValueError(
            "No valid images found in the dataset!"
        )

    X = np.array(
        features,
        dtype=np.float32
    )

    y = np.array(
        labels,
        dtype=np.int32
    )

    feature_names = generate_feature_names()

    if X.shape[1] != len(feature_names):

        raise ValueError(
            "Feature name count does not match feature vector!"
        )

    # Save extracted dataset
    feature_path = os.path.join(
        RESULTS_DIR,
        "extracted_features.pkl"
    )

    with open(feature_path, "wb") as f:

        pickle.dump(
            (X, y),
            f
        )

    # Save feature names for visualization
    names_path = os.path.join(
        RESULTS_DIR,
        "feature_names.pkl"
    )

    with open(names_path, "wb") as f:

        pickle.dump(
            feature_names,
            f
        )

    # Save extraction configuration
    config = {
        "classes": CLASSES,
        "image_size": IMG_SIZE,
        "feature_count": X.shape[1],
        "features": [
            "HSV Histogram",
            "RGB Histogram",
            "RGB Statistics",
            "HOG",
            "LBP",
            "Edge Density"
        ]
    }

    with open(
        os.path.join(RESULTS_DIR, "feature_config.pkl"),
        "wb"
    ) as f:

        pickle.dump(config, f)

    # ======================================
    # FINAL SUMMARY
    # ======================================

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETED")
    print("=" * 60)

    print("Total images:", len(X))
    print("Total features per image:", X.shape[1])

    print("\nClass distribution:")

    for index, class_name in enumerate(CLASSES):

        count = np.sum(y == index)

        print(
            f"{class_name:12}: {count}"
        )

    print("\nFeature types:")
    print("HSV Histogram       : 512")
    print("RGB Histogram       : 24")
    print("RGB Statistics      : 6")
    print("HOG Features        : 576")
    print("LBP Features        : 256")
    print("Edge Density        : 1")

    print("\nTotal:", X.shape[1])

    print("\nSaved files:")
    print(feature_path)
    print(names_path)
    print("./results/feature_config.pkl")
    print("=" * 60)


# ==========================================
# 11. MAIN
# ==========================================

if __name__ == "__main__":

    process_dataset()
