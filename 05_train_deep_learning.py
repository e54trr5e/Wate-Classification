
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.utils.class_weight import compute_class_weight


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_DIR = "./dataset"
MODEL_DIR = "./models"
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
EPOCHS = 40
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)


# ==========================================
# 2. LOAD DATASET
# ==========================================

def load_dataset():

    print("\nLoading image dataset...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.20,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_names=CLASSES,
        label_mode="int"
    )

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

    print("\nClass names:")
    print(train_ds.class_names)

    # Prefetch for better performance
    train_ds = train_ds.prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    val_ds = val_ds.prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    return train_ds, val_ds


# ==========================================
# 3. DATA AUGMENTATION
# ==========================================

def create_augmentation():

    augmentation = tf.keras.Sequential([

        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.10
        ),

        tf.keras.layers.RandomZoom(
            0.15
        ),

        tf.keras.layers.RandomContrast(
            0.15
        ),

        tf.keras.layers.RandomTranslation(
            0.10,
            0.10
        )

    ], name="data_augmentation")

    return augmentation


# ==========================================
# 4. BUILD CNN MODEL
# ==========================================

def build_cnn():

    augmentation = create_augmentation()

    inputs = tf.keras.Input(
        shape=(180, 180, 3)
    )

    # Data augmentation
    x = augmentation(inputs)

    # Normalize pixel values
    x = tf.keras.layers.Rescaling(
        1.0 / 255
    )(x)

    # Convolution block 1
    x = tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    # Convolution block 2
    x = tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    # Convolution block 3
    x = tf.keras.layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    # Convolution block 4
    x = tf.keras.layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    # Feature aggregation
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Fully connected layer
    x = tf.keras.layers.Dense(
        256,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(
        0.5
    )(x)

    # Six output categories
    outputs = tf.keras.layers.Dense(
        len(CLASSES),
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Smart_Waste_CNN"
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    return model


# ==========================================
# 5. CALCULATE CLASS WEIGHTS
# ==========================================

def calculate_class_weights(train_ds):

    labels = np.concatenate([
        labels.numpy()
        for _, labels in train_ds
    ])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASSES)),
        y=labels
    )

    class_weights = {
        i: float(weight)
        for i, weight in enumerate(weights)
    }

    print("\nClass weights:")
    print(class_weights)

    return class_weights


# ==========================================
# 6. TRAIN CNN
# ==========================================

def train_model(model, train_ds, val_ds):

    print("\nStarting CNN training...")

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                MODEL_DIR,
                "best_cnn_model.keras"
            ),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )

    ]

    class_weights = calculate_class_weights(
        train_ds
    )

    history = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=EPOCHS,

        callbacks=callbacks,

        class_weight=class_weights

    )

    return history


# ==========================================
# 7. EVALUATE CNN
# ==========================================

def evaluate_model(model, val_ds):

    print("\nEvaluating CNN...")

    y_true = np.concatenate([
        labels.numpy()
        for _, labels in val_ds
    ])

    probabilities = model.predict(
        val_ds,
        verbose=1
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("CNN EVALUATION RESULTS")
    print("=" * 60)

    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1 Score : {f1 * 100:.2f}%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(CLASSES)),
            target_names=CLASSES,
            zero_division=0
        )
    )

    return y_true, y_pred, {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }


# ==========================================
# 8. GENERATE TRAINING GRAPHS
# ==========================================

def plot_training_history(history):

    # Accuracy graph
    plt.figure(figsize=(9, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title("CNN Training and Validation Accuracy")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "cnn_accuracy.png"
        ),
        dpi=250
    )

    plt.close()

    # Loss graph
    plt.figure(figsize=(9, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title("CNN Training and Validation Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "cnn_loss.png"
        ),
        dpi=250
    )

    plt.close()

    print("\nTraining graphs saved.")


# ==========================================
# 9. CONFUSION MATRIX
# ==========================================

def save_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(CLASSES))
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

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
        "CNN Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "cnn_confusion_matrix.png"
        ),
        dpi=250
    )

    plt.close()

    print("CNN confusion matrix saved.")


# ==========================================
# 10. SAVE FINAL MODEL AND RESULTS
# ==========================================

def save_results(model, metrics):

    model_path = os.path.join(
        MODEL_DIR,
        "cnn_waste_model.keras"
    )

    model.save(model_path)

    # Save class names and image dimensions
    metadata = {
        "classes": CLASSES,
        "image_size": IMG_SIZE,
        "model_type": "CNN",
        "metrics": metrics
    }

    with open(
        os.path.join(
            MODEL_DIR,
            "cnn_metadata.json"
        ),
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    with open(
        os.path.join(
            RESULTS_DIR,
            "cnn_results.json"
        ),
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print("\nSaved CNN model:", model_path)


# ==========================================
# 11. MAIN TRAINING PIPELINE
# ==========================================

def main():

    print("\n" + "=" * 60)
    print("SMART WASTE CLASSIFICATION")
    print("ADVANCED CNN TRAINING")
    print("=" * 60)

    # Load dataset
    train_ds, val_ds = load_dataset()

    # Build model
    model = build_cnn()

    print("\nCNN Model Architecture:")

    model.summary()

    # Train model
    history = train_model(
        model,
        train_ds,
        val_ds
    )

    # Load best checkpoint
    best_path = os.path.join(
        MODEL_DIR,
        "best_cnn_model.keras"
    )

    best_model = tf.keras.models.load_model(
        best_path
    )

    # Evaluate best checkpoint
    y_true, y_pred, metrics = evaluate_model(
        best_model,
        val_ds
    )

    # Generate graphs
    plot_training_history(history)

    # Generate confusion matrix
    save_confusion_matrix(
        y_true,
        y_pred
    )

    # Save final model
    save_results(
        best_model,
        metrics
    )

    print("\n" + "=" * 60)
    print("CNN TRAINING COMPLETED")
    print("=" * 60)

    print(
        "CNN Validation Accuracy:",
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print("Model saved successfully.")

    print("=" * 60)


# ==========================================
# 12. EXECUTION
# ==========================================

if __name__ == "__main__":

    main()
