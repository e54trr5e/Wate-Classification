
import os
import sys
import numpy as np
import tensorflow as tf

# Model location
MODEL_PATH = "./models/best_cnn_model.keras"

# Must match the class order used during CNN training
CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

IMAGE_SIZE = (180, 180)


def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print("Loading trained CNN model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("CNN model loaded successfully!")
    return model


def predict_waste(model, image_path):

    if not os.path.exists(image_path):
        print("Error: Image not found:", image_path)
        return

    # Load and resize image
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    # Convert image into a NumPy array
    image_array = tf.keras.utils.img_to_array(image)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    predictions = model.predict(image_array, verbose=0)[0]

    # Get top 3 predictions
    top_indices = np.argsort(predictions)[::-1][:3]

    print("\n" + "=" * 40)
    print("       WASTE CLASSIFICATION")
    print("=" * 40)

    predicted_index = top_indices[0]
    predicted_class = CLASSES[predicted_index]
    confidence = predictions[predicted_index] * 100

    print("\nPredicted Category:", predicted_class.upper())
    print("Prediction Score: {:.2f}%".format(confidence))

    print("\nTop 3 Predictions:")
    for index in top_indices:
        print(
            "{}: {:.2f}%".format(
                CLASSES[index],
                predictions[index] * 100
            )
        )

    print("\n" + "=" * 40)


if __name__ == "__main__":

    try:
        model = load_trained_model()

        if len(sys.argv) > 1:
            image_path = sys.argv[1]
        else:
            image_path = input(
                "\nEnter the waste image path: "
            ).strip().strip('"')

        predict_waste(model, image_path)

    except Exception as error:
        print("\nError:", error)
