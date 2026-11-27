import cv2
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# ==========================================
# 1. LOAD YOLO MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "yolo_classification"
    / "waste_yolov8n_cls"
    / "weights"
    / "best.pt"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print("Loading YOLOv8 model...")
model = YOLO(str(MODEL_PATH))
print("Model loaded successfully!")


# ==========================================
# 2. WASTE BIN MAPPING & COUNTERS
# ==========================================

bin_mapping = {
    "cardboard": "Paper/Cardboard Bin",
    "glass": "Glass Recycling Bin",
    "metal": "Metal Recycling Bin",
    "paper": "Paper Recycling Bin",
    "plastic": "Plastic Recycling Bin",
    "trash": "General Waste Bin"
}

waste_counts = defaultdict(int)
total_items = 0


# ==========================================
# 3. START WEBCAM
# ==========================================

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Unable to open webcam.")

print("\nWebcam started!")
print("Position waste item in front of the camera.")
print("SPACE = Count item | Q = Quit\n")

prev_time = time.time()


# ==========================================
# 4. REAL-TIME CLASSIFICATION LOOP
# ==========================================

try:
    while True:
        success, frame = camera.read()
        if not success:
            print("Camera frame unavailable.")
            break

        h, w, _ = frame.shape
        display_frame = frame.copy()

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
        prev_time = curr_time

        # --------------------------------------------------
        # DYNAMIC ROI LOCALIZATION (COLOR MASK + CONTOURS)
        # --------------------------------------------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Color mask tuned for cyan/blue plastic bottles & active foreground items
        lower_blue = np.array([75, 30, 30])
        upper_blue = np.array([135, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Clean noise around contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_cleaned = cv2.dilate(mask_cleaned, kernel, iterations=2)

        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 1200]

        if valid_contours:
            # Tight box directly over the bottle
            largest_contour = max(valid_contours, key=cv2.contourArea)
            rx, ry, rw, rh = cv2.boundingRect(largest_contour)

            pad = 15
            xmin = max(0, rx - pad)
            ymin = max(0, ry - pad)
            xmax = min(w, rx + rw + pad)
            ymax = min(h, ry + rh + pad)

            roi = frame[ymin:ymax, xmin:xmax]
        else:
            # Fallback center region box
            box_w, box_h = int(w * 0.35), int(h * 0.55)
            xmin = (w - box_w) // 2
            ymin = (h - box_h) // 2
            xmax = xmin + box_w
            ymax = ymin + box_h
            roi = frame[ymin:ymax, xmin:xmax]

        # Run model inference on the isolated crop
        results = model.predict(source=roi, imgsz=224, verbose=False)
        result = results[0]

        if result.probs is not None:
            class_id = result.probs.top1
            confidence = float(result.probs.top1conf) * 100
            waste_type = model.names[class_id].lower()

            recommended_bin = bin_mapping.get(waste_type, "General Waste Bin")
        else:
            waste_type = "Unknown"
            confidence = 0.0
            recommended_bin = "Scanning..."

        # --------------------------------------------------
        # DRAW BOUNDING BOX & HUD
        # --------------------------------------------------
        cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

        label_text = f"{waste_type.upper()} ({confidence:.1f}%)"
        cv2.rectangle(
            display_frame,
            (xmin, max(0, ymin - 35)),
            (xmin + len(label_text) * 16 + 15, ymin),
            (0, 255, 0),
            -1
        )
        cv2.putText(
            display_frame,
            label_text,
            (xmin + 8, max(20, ymin - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

        cv2.rectangle(display_frame, (10, 10), (520, 130), (0, 0, 0), -1)

        cv2.putText(
            display_frame,
            f"Waste: {waste_type.upper()}",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display_frame,
            f"Bin: {recommended_bin}",
            (25, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"FPS: {fps:.1f} | Total Counted: {total_items}",
            (25, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.putText(
            display_frame,
            "SPACE: Count Item | Q: Quit",
            (20, display_frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow("AI Smart Waste Segregation System", display_frame)

        # --------------------------------------------------
        # CONTROLS
        # --------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if waste_type != "Unknown":
                waste_counts[waste_type] += 1
                total_items += 1
                print(f"Counted: {waste_type.upper()} | Total Items: {total_items}")
        elif key == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()


# ==========================================
# 5. FINAL REPORT
# ==========================================

print("\n" + "=" * 50)
print("       WASTE SEGREGATION REPORT")
print("=" * 50)
print(f"Total Items Counted: {total_items}\n")

for waste_type in bin_mapping:
    print(f"{waste_type.capitalize():<15}: {waste_counts[waste_type]}")

print("\nRecommended Bin Summary:")
for waste_type, count in waste_counts.items():
    if count > 0:
        print(f"-> {bin_mapping[waste_type]}: {count}")

print("\nDemonstration completed successfully!")