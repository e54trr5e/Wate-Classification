# AI-Based Smart Waste Classification

An image-classification project that predicts one of six waste categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

The repository explores traditional machine learning with handcrafted image features, a custom CNN, and Ultralytics YOLOv8 image classification. It also includes a webcam demonstration.

> **Task distinction:** This dataset is organized by one class per image. The YOLO workflow in this repository is **image classification**, not object detection. Object detection requires bounding-box annotations and a detector trained for that task.

## Current project components

| File | Purpose |
|---|---|
| `01_analyze_dataset.py` | Summarizes and plots the dataset class distribution |
| `02_extract_features.py` | Extracts HSV/RGB histograms, color statistics, HOG, LBP and edge density |
| `03_train_ml.py` | Trains Random Forest, KNN, Logistic Regression and SVM; selects the best model by macro F1 |
| `04_predict_ml.py` | Predicts a class for a single image using the saved traditional ML model |
| `05_train_deep_learning.py` | Trains and evaluates a custom CNN |
| `06_train_yolo.py` | Retired experimental NumPy script; **not YOLO** and must not be used for YOLO results |
| `07_plot_results.py` | Plots available YOLO training curves from its generated CSV |
| `08_live_demo.py` | Webcam-based single-item classification using the trained YOLO classification model |
| `09_prepare_yolo_classification.py` | Creates class-stratified train/validation/test folders |
| `09_train_yolo_classification.py` | Trains Ultralytics YOLOv8n image classification |
| `10_evaluate_yolo.py` | Evaluates YOLOv8 classification on the held-out test folder |
| `final_model_comparison.py` | Reads saved model metrics; compare models only after using a common split and protocol |

## Dataset layout

Place the original images into one folder per class:

```text
dataset/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
```

The scripts expect these exact lowercase class-folder names.

## Environment setup

Use Python 3.10 or 3.11 for the best compatibility with the computer-vision and deep-learning packages. Create and activate a virtual environment, then install dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow and PyTorch/Ultralytics can have platform-specific installation requirements. If installation fails, install the compatible build for your operating system and Python version using the official package instructions.

## Traditional machine learning workflow

```powershell
python 01_analyze_dataset.py
python 02_extract_features.py
python 03_train_ml.py
python 04_predict_ml.py .\path\to\image.jpg
```

Feature extraction writes files under `results/`. Training writes model files under `models/`.

## CNN workflow

```powershell
python 05_train_deep_learning.py
```

The current CNN script creates its own seeded train/validation split and evaluates on validation data. Its metrics should not be presented as directly comparable to the traditional ML or YOLO test metrics until all model families are migrated to the same fixed split.

## YOLOv8 image-classification workflow

First prepare the class-stratified folders, then train and evaluate:

```powershell
python 09_prepare_yolo_classification.py
python 09_train_yolo_classification.py
python 10_evaluate_yolo.py
```

The prepared folder structure is:

```text
yolo_classification/
  train/<class_name>/
  val/<class_name>/
  test/<class_name>/
```

The split uses a fixed seed. If the output folder already exists, the preparation script stops rather than silently overwriting it. Delete or rename that folder only when you intentionally want to regenerate the split.

## Webcam demonstration

After training the YOLO classification model and confirming the model path in `08_live_demo.py`:

```powershell
python 08_live_demo.py
```

The demo classifies a selected camera region and displays the model's returned class and confidence. The colored contour logic is only used to propose a region of interest; it must not override the model's class or confidence. Press **SPACE** to count the current prediction and **Q** to quit.

## Evaluation and reporting notes

- Report accuracy together with macro F1, weighted F1, per-class precision/recall/F1 and a confusion matrix.
- Macro F1 is important because it gives each class equal weight, including the smaller Trash class.
- Never compare validation metrics from one model with test metrics from another as if they came from one benchmark.
- The traditional ML script currently creates a reproducible stratified 80/20 split. The CNN and YOLO workflows currently use their own splits; a shared manifest/split is still needed for a fair cross-family comparison.
- Do not quote old accuracy numbers from plots or reports unless they are regenerated from the current saved evaluation outputs.
- Prediction confidence is a model score, not a guarantee that the prediction is correct. Any acceptance threshold should be selected using validation data and documented.

## Current limitations

- `06_train_yolo.py` is deliberately retired. It is not a genuine YOLO implementation.
- The custom CNN is trained from scratch; MobileNetV2 transfer learning is a planned improvement, not yet implemented.
- Grad-CAM, calibrated confidence thresholds, SQLite prediction analytics and a unified shared split are future improvements unless separately implemented.
- This repository's image-level dataset does not support claims of multi-object localization or bounding-box detection.

## Team

| Name | GitHub | Contribution |
|---|---|---|
| Mehakdeep Kaur | [@Mehak-009deep](https://github.com/Mehak-009deep) | Coding, model testing, presentation and report |
| Krrish | [@e54trr5e](https://github.com/e54trr5e) | Coding and model training |
