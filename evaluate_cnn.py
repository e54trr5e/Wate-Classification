
import os
import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Configuration
DATASET_DIR = "./dataset"
MODEL_PATH = "./models/best_cnn_model.keras"
RESULTS_DIR = "./results"

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

IMG_SIZE = (180, 180)
BATCH_SIZE = 32
SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)

# Check model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"CNN model not found: {MODEL_PATH}"
    )

print("Loading saved CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

# Recreate the same validation split used during training
print("Loading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=CLASSES,
    label_mode="int",
    shuffle=False
)

# Actual labels
y_true = np.concatenate([
    labels.numpy()
    for images, labels in val_ds
])

# Model predictions
print("Predicting images...")

probabilities = model.predict(
    val_ds,
    verbose=1
)

y_pred = np.argmax(probabilities, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true, y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true, y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true, y_pred,
    average="weighted",
    zero_division=0
)

# Display results
print("\n========== CNN EVALUATION ==========")

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")

print("\nClassification Report:")

print(classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(CLASSES)),
    target_names=CLASSES,
    zero_division=0
))

# Save metrics
metrics = {
    "model": "CNN",
    "evaluation_split": "validation",
    "accuracy": round(float(accuracy * 100), 2),
    "precision": round(float(precision * 100), 2),
    "recall": round(float(recall * 100), 2),
    "f1_score": round(float(f1 * 100), 2)
}

output_path = os.path.join(
    RESULTS_DIR,
    "cnn_results.json"
)

with open(output_path, "w") as file:
    json.dump(metrics, file, indent=4)

print("\nCNN metrics saved to:", output_path)
