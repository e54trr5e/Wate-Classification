import os
import cv2
import pickle
import numpy as np

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE = (64, 64)
GRID_SIZE = 4  # 4x4 spatial detection grid (YOLO style)
DATASET_DIR = "./dataset"

class SimpleYOLOv8Detector:
    """
    A lightweight, pure-NumPy implementation of a YOLO-style object detector.
    Divides images into a S x S spatial grid and outputs bounding boxes, 
    objectness confidence scores, and class probabilities per grid cell.
    """
    def __init__(self, grid_s=GRID_SIZE, num_classes=len(CLASSES)):
        self.grid_s = grid_s
        self.num_classes = num_classes
        # Output vector per cell: [x_offset, y_offset, w, h, confidence, prob_0..prob_5]
        self.cell_out_dim = 5 + num_classes
        self.input_dim = IMG_SIZE[0] * IMG_SIZE[1] * 3
        self.output_dim = grid_s * grid_s * self.cell_out_dim

        np.random.seed(42)
        self.W = np.random.randn(self.input_dim, self.output_dim) * 0.01
        self.b = np.zeros((1, self.output_dim))

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))

    def forward(self, X):
        raw_out = np.dot(X, self.W) + self.b
        # Reshape to (Batch, Grid, Grid, Cell_Out)
        batch_size = X.shape[0]
        grid_out = raw_out.reshape(batch_size, self.grid_s, self.grid_s, self.cell_out_dim)
        
        # Apply Sigmoid to bounding box parameters, confidence, and class probabilities
        grid_out = self._sigmoid(grid_out)
        return grid_out

    def fit(self, X, y, epochs=15, lr=0.01, batch_size=32):
        n_samples = X.shape[0]
        print(f"Training YOLO Object Detection Model ({self.grid_s}x{self.grid_s} Grid)...")

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, batch_size):
                X_b = X_shuffled[i:i+batch_size]
                y_b = y_shuffled[i:i+batch_size]
                b_size = X_b.shape[0]

                # Create ground truth YOLO targets
                targets = np.zeros((b_size, self.grid_s, self.grid_s, self.cell_out_dim))
                for b_idx in range(b_size):
                    cls_idx = y_b[b_idx]
                    # Assign object to center cell (2, 2)
                    center_cell = self.grid_s // 2
                    targets[b_idx, center_cell, center_cell, 0:4] = 0.5  # Box center/size
                    targets[b_idx, center_cell, center_cell, 4] = 1.0    # Objectness confidence = 1
                    targets[b_idx, center_cell, center_cell, 5 + cls_idx] = 1.0  # Class target

                # Forward Pass
                raw_out = np.dot(X_b, self.W) + self.b
                preds = self._sigmoid(raw_out.reshape(b_size, self.grid_s, self.grid_s, self.cell_out_dim))

                # Gradient Computation (MSE Loss on Bounding Boxes + Confidence + Classes)
                grad_grid = (preds - targets) / b_size
                grad_raw = grad_grid.reshape(b_size, -1)

                # Backward Pass & Parameter Update
                dW = np.dot(X_b.T, grad_raw)
                db = np.sum(grad_raw, axis=0, keepdims=True)

                self.W -= lr * dW
                self.b -= lr * db

            print(f"Epoch {epoch+1}/{epochs} | YOLO Loss: {np.mean(np.square(preds - targets)):.4f}")

    def detect(self, img_flat):
        grid_pred = self.forward(img_flat.reshape(1, -1))[0]
        
        # Find cell with highest objectness confidence
        conf_map = grid_pred[:, :, 4]
        max_cell = np.unravel_index(np.argmax(conf_map), conf_map.shape)
        cell_y, cell_x = max_cell
        
        cell_data = grid_pred[cell_y, cell_x]
        confidence = cell_data[4]
        class_probs = cell_data[5:]
        pred_class_idx = np.argmax(class_probs)
        
        return {
            'grid_cell': (cell_x, cell_y),
            'confidence': confidence,
            'class_idx': pred_class_idx,
            'class_name': CLASSES[pred_class_idx]
        }


def load_dataset():
    X, y = [], []
    print("Loading image frames for YOLO object detection...")
    for label_idx, c in enumerate(CLASSES):
        class_path = os.path.join(DATASET_DIR, c)
        if not os.path.exists(class_path):
            continue
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, IMG_SIZE)
                X.append(img.flatten() / 255.0)
                y.append(label_idx)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = load_dataset()

    yolo_model = SimpleYOLOv8Detector(grid_s=GRID_SIZE, num_classes=len(CLASSES))
    yolo_model.fit(X, y, epochs=15, lr=0.01)

    # Test detection on sample item
    test_sample = X[0]
    result = yolo_model.detect(test_sample)

    print("\n" + "=" * 55)
    print("YOLO OBJECT DETECTION RESULT")
    print("=" * 55)
    print(f"Detected Object : {result['class_name'].upper()}")
    print(f"Grid Location   : Cell {result['grid_cell']}")
    print(f"Confidence Score: {result['confidence'] * 100:.2f}%")
    print("=" * 55)

    os.makedirs("./models", exist_ok=True)
    save_path = "./models/yolo_waste_detector.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(yolo_model, f)
    print(f"\nSaved YOLO model weights to {save_path}")