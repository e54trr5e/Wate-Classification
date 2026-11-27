
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

FEATURE_FILE = "./results/extracted_features.pkl"
FEATURE_NAMES_FILE = "./results/feature_names.pkl"

MODEL_DIR = "./models"
RESULTS_DIR = "./results"

RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

def load_dataset():

    if not os.path.exists(FEATURE_FILE):
        raise FileNotFoundError(
            "Features not found! Run 02_extract_features.py first."
        )

    with open(FEATURE_FILE, "rb") as f:
        X, y = pickle.load(f)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    if X.ndim != 2:
        raise ValueError("Feature dataset must be two-dimensional.")

    if len(X) != len(y) or len(X) == 0:
        raise ValueError("Invalid or empty dataset.")

    if not np.isfinite(X).all():
        raise ValueError("Dataset contains NaN or infinite values.")

    if not np.isin(y, np.arange(len(CLASSES))).all():
        raise ValueError("Dataset contains unknown class labels.")

    print("\nDATASET LOADED SUCCESSFULLY")
    print("-" * 45)
    print("Total images:", len(X))
    print("Features per image:", X.shape[1])

    for index, name in enumerate(CLASSES):
        print(f"{name:12}: {np.sum(y == index)}")

    return X, y


# ==========================================
# 3. DEFINE MACHINE LEARNING MODELS
# ==========================================

def create_models():

    models = {

        "random_forest_model.pkl": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "k-nearest_neighbors_model.pkl": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                metric="minkowski"
            ))
        ]),

        "logistic_regression_model.pkl": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),

        "support_vector_machine_model.pkl": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ])
    }

    return models


# ==========================================
# 4. SAVE MODEL AND METADATA
# ==========================================

def save_model(model, filename, accuracy):

    model_path = os.path.join(
        MODEL_DIR,
        filename
    )

    bundle = {
        "model": model,
        "classes": CLASSES,
        "feature_count": 1375,
        "feature_extraction": [
            "HSV Histogram",
            "RGB Histogram",
            "RGB Statistics",
            "HOG",
            "LBP",
            "Edge Density"
        ],
        "test_accuracy": float(accuracy),
        "random_state": RANDOM_STATE
    }

    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    print("Model saved:", model_path)


# ==========================================
# 5. EVALUATE MODEL
# ==========================================

def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("MODEL:", name)
    print("=" * 60)

    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"Weighted F1: {f1 * 100:.2f}%")
    print(f"Macro F1   : {macro_f1 * 100:.2f}%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=np.arange(len(CLASSES)),
            target_names=CLASSES,
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "macro_f1": macro_f1,
        "predictions": predictions
    }


# ==========================================
# 6. GENERATE CONFUSION MATRIX
# ==========================================

def save_confusion_matrix(name, y_test, predictions):

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(CLASSES))
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASSES
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=35
    )

    ax.set_title(
        f"{name} Confusion Matrix"
    )

    plt.tight_layout()

    filename = (
        name.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        os.path.join(RESULTS_DIR, filename),
        dpi=250
    )

    plt.close()

    print("Confusion matrix saved:", filename)


# ==========================================
# 7. TRAIN ALL MODELS
# ==========================================

def train_all_models():

    X, y = load_dataset()

    counts = np.bincount(
        y,
        minlength=len(CLASSES)
    )

    if np.any(counts < 2):
        raise ValueError(
            "Every category needs at least two images."
        )

    if len(X) * TEST_SIZE < len(CLASSES):
        raise ValueError(
            "Not enough samples for a stratified test set. "
            "Add more images."
        )

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTRAINING DATA:", len(X_train))
    print("TESTING DATA :", len(X_test))

    models = create_models()

    results = []

    best_macro_f1 = -1
    best_accuracy = None
    best_model = None
    best_model_name = None

    print("\n" + "=" * 60)
    print("SMART WASTE CLASSIFICATION")
    print("ADVANCED MACHINE LEARNING TRAINING")
    print("=" * 60)

    # ======================================
    # TRAIN EACH ALGORITHM
    # ======================================

    for filename, model in models.items():

        model_name = filename.replace(
            "_model.pkl", ""
        ).replace("_", " ").title()

        print("\nTraining:", model_name)

        model.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            model_name,
            model,
            X_test,
            y_test
        )

        # Save individual model
        save_model(
            model,
            filename,
            metrics["accuracy"]
        )

        # Save confusion matrix
        save_confusion_matrix(
            model_name,
            y_test,
            metrics["predictions"]
        )

        results.append({
            "name": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "macro_f1": metrics["macro_f1"]
        })

        if metrics["macro_f1"] > best_macro_f1:

            best_macro_f1 = metrics["macro_f1"]
            best_accuracy = metrics["accuracy"]
            best_model = model
            best_model_name = model_name

    # ======================================
    # SAVE BEST MODEL
    # ======================================

    best_bundle = {
        "model": best_model,
        "classes": CLASSES,
        "feature_count": X.shape[1],
        "feature_extraction": [
            "HSV Histogram",
            "RGB Histogram",
            "RGB Statistics",
            "HOG",
            "LBP",
            "Edge Density"
        ],
        "test_accuracy": float(best_accuracy),
        "test_macro_f1": float(best_macro_f1),
        "model_name": best_model_name,
        "random_state": RANDOM_STATE
    }

    best_path = os.path.join(
        MODEL_DIR,
        "best_ml_model.pkl"
    )

    with open(best_path, "wb") as f:
        pickle.dump(best_bundle, f)

    # ======================================
    # FINAL RESULTS
    # ======================================

    print("\n" + "=" * 75)
    print("FINAL MACHINE LEARNING MODEL COMPARISON")
    print("=" * 75)

    print(
        f"{'MODEL':30}"
        f"{'ACCURACY':12}"
        f"{'PRECISION':12}"
        f"{'RECALL':12}"
        f"{'F1 SCORE':12}"
    )

    print("-" * 75)

    for result in sorted(
        results,
        key=lambda x: x["macro_f1"],
        reverse=True
    ):

        print(
            f"{result['name']:30}"
            f"{result['accuracy'] * 100:8.2f}%"
            f"{result['precision'] * 100:9.2f}%"
            f"{result['recall'] * 100:9.2f}%"
            f"{result['f1_score'] * 100:9.2f}%"
        )

    print("-" * 75)

    print("\nBEST MODEL:", best_model_name)
    print("BEST MACRO F1:", round(best_macro_f1 * 100, 2), "%")
    print("BEST MODEL ACCURACY:", round(best_accuracy * 100, 2), "%")
    print("SAVED TO:", best_path)

    # ======================================
    # SAVE RESULTS
    # ======================================

    with open(
        os.path.join(RESULTS_DIR, "ml_model_results.pkl"),
        "wb"
    ) as f:

        pickle.dump(results, f)

    print("\nAll four ML models trained successfully!")
    print("=" * 75)


# ==========================================
# 8. MAIN
# ==========================================

if __name__ == "__main__":

    train_all_models()
