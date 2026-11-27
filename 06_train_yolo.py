"""DEPRECATED: This is not a YOLO implementation.

This former script implemented a small NumPy classifier/detector without
bounding-box annotations or a real YOLO architecture. It must not be used
or reported as YOLO/YOLOv8 object detection.

For this image-level six-class waste dataset, use:
    python 09_prepare_yolo_classification.py
    python 09_train_yolo_classification.py
    python 10_evaluate_yolo.py

Use a genuine object-detection dataset with bounding-box annotations if you
need to train an object detector.
"""

if __name__ == "__main__":
    raise SystemExit(
        "This experimental NumPy detector has been retired. "
        "Use 09_train_yolo_classification.py for YOLOv8 image classification."
    )
