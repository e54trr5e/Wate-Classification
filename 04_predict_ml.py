
import os
import sys
import pickle
import cv2
import numpy as np
import importlib.util


# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_PATH = "./models/best_ml_model.pkl"

FEATURE_SCRIPT = "./02_extract_features.py"

CONFIDENCE_THRESHOLD = 50.0


# ==========================================
# 2. LOAD FEATURE EXTRACTION FUNCTION
# ==========================================

def load_feature_extractor():

    spec = importlib.util.spec_from_file_location(
        "feature_extractor",
        FEATURE_SCRIPT
    )

    if spec is None or spec.loader is None:
        raise ImportError("Cannot load feature extraction script.")

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module.extract_features


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found! Run 03_train_ml.py first."
        )

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    # New model format
    if isinstance(bundle, dict):

        model = bundle["model"]
        classes = bundle["classes"]
        expected_features = bundle["feature_count"]

        print("\nMODEL INFORMATION")
        print("-" * 45)
        print("Model:", bundle.get("model_name", "Best ML Model"))
        print(
            "Test Accuracy:",
            f"{bundle['test_accuracy'] * 100:.2f}%"
        )
        print("Feature Count:", expected_features)

    else:
        # Legacy model format
        model = bundle
        classes = [
            "cardboard", "glass", "metal",
            "paper", "plastic", "trash"
        ]
        expected_features = 1375

    return model, classes, expected_features


# ==========================================
# 4. PREDICT WASTE CATEGORY
# ==========================================

def predict_image(image_path):

    print("\n" + "=" * 55)
    print("SMART WASTE CLASSIFICATION - IMAGE PREDICTION")
    print("=" * 55)

    # Load model
    model, classes, expected_features = load_model()

    # Load feature extraction function
    extract_features = load_feature_extractor()

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            "Could not read image. Check the image path."
        )

    print("\nImage:", image_path)

    # Extract features
    features = extract_features(image)

    features = features.reshape(1, -1)

    print("Extracted Features:", features.shape[1])

    # Verify feature compatibility
    if features.shape[1] != expected_features:

        raise ValueError(
            f"Feature mismatch! Model expects "
            f"{expected_features}, but received "
            f"{features.shape[1]}."
        )

    # Predict category
    prediction = model.predict(features)[0]

    predicted_class = classes[int(prediction)]

    # Get probability estimates
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(features)[0]

        model_classes = model.classes_

        # Match probability indices to class labels
        probability_map = {
            int(class_index): float(probability)
            for class_index, probability
            in zip(model_classes, probabilities)
        }

        class_probabilities = [
            probability_map.get(i, 0.0)
            for i in range(len(classes))
        ]

    else:

        raise RuntimeError(
            "This model does not support probability prediction."
        )

    # Sort top three predictions
    top_indices = np.argsort(
        class_probabilities
    )[::-1][:3]

    top_confidence = (
        class_probabilities[int(top_indices[0])] * 100
    )

    # ======================================
    # 5. DISPLAY RESULTS
    # ======================================

    print("\n" + "=" * 55)
    print("PREDICTION RESULTS")
    print("=" * 55)

    print("\nPredicted Category:", predicted_class.upper())

    print("\nTOP 3 PREDICTIONS")
    print("-" * 40)

    for rank, index in enumerate(top_indices, start=1):

        category = classes[int(index)]

        confidence = (
            class_probabilities[int(index)] * 100
        )

        print(
            f"{rank}. {category.upper():12} "
            f"| {confidence:.2f}%"
        )

    # ======================================
    # 6. CONFIDENCE WARNING
    # ======================================

    print("\n" + "-" * 40)

    if top_confidence >= CONFIDENCE_THRESHOLD:

        print("STATUS: HIGHER CONFIDENCE PREDICTION")

        print(
            "Predicted Category:",
            predicted_class.upper()
        )

    else:

        print("STATUS: LOW CONFIDENCE")
        print("Prediction requires manual verification.")

    print("-" * 40)

    # ======================================
    # 7. DISPLAY INPUT IMAGE
    # ======================================

    display_image = cv2.resize(
        image,
        (500, 500)
    )

    label = (
        f"{predicted_class.upper()} "
        f"| {top_confidence:.1f}%"
    )

    cv2.putText(
        display_image,
        label,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if top_confidence >= CONFIDENCE_THRESHOLD
        else (0, 165, 255),
        2
    )

    cv2.imshow(
        "Smart Waste Classification",
        display_image
    )

    print("\nPress any key to close the image.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nPrediction completed successfully!")


# ==========================================
# 8. MAIN FUNCTION
# ==========================================

if __name__ == "__main__":

    try:

        if len(sys.argv) > 1:

            image_path = sys.argv[1]

        else:

            image_path = input(
                "\nEnter image path: "
            ).strip().strip('"')

        if not os.path.exists(image_path):

            raise FileNotFoundError(
                "Image path does not exist!"
            )

        predict_image(image_path)

    except Exception as error:

        print("\nERROR:", error)

    finally:

        cv2.destroyAllWindows()
