# AI-Based Smart Waste Classification

An image-classification system for six waste categories: **cardboard, glass, metal, paper, plastic and trash**.

The project compares handcrafted-feature machine learning, a custom CNN, MobileNetV2 transfer learning and YOLOv8 image classification. It includes Grad-CAM explanations, validation-based confidence review, a webcam demonstration and a Streamlit application with SQLite prediction history.

> **Important terminology:** The dataset contains one class label per image. YOLOv8 is used here for **image classification**, not object detection. Object detection requires bounding-box annotations and a detector trained on those annotations.

## Project architecture

| Component | Script | Purpose |
|---|---|---|
| Shared dataset | `00_prepare_shared_dataset.py` | Creates one reproducible stratified 70/15/15 split and a CSV manifest |
| Traditional features | `02_extract_features.py`, `02_extract_shared_features.py` | HSV/RGB histograms, color statistics, HOG, LBP and edge density |
| Traditional ML | `03_train_ml.py` | Random Forest, KNN, Logistic Regression and SVM; selection by validation macro-F1 |
| ML prediction | `04_predict_ml.py` | Single-image inference with validation-calibrated manual-review threshold |
| Custom CNN | `05_train_deep_learning.py` | CNN trained and tested on the shared split |
| Retired experiment | `06_train_yolo.py` | Retired custom NumPy experiment; not a YOLO implementation |
| Training plots | `07_plot_results.py` | Plots available Ultralytics training curves; does not hardcode benchmark metrics |
| Webcam | `08_live_demo.py` | YOLOv8 classification on a webcam ROI; model prediction is not overwritten by color rules |
| Legacy dataset command | `09_prepare_yolo_classification.py` | Compatibility wrapper that calls the common split generator |
| YOLOv8 classification | `09_train_yolo_classification.py`, `10_evaluate_yolo.py` | Trains and evaluates YOLOv8n classification on the shared folders |
| YOLO prediction | `11_predict_yolo.py` | Single-image YOLOv8 classification |
| Transfer learning | `12_train_mobilenetv2.py` | MobileNetV2 ImageNet transfer learning and upper-layer fine-tuning |
| Explainability | `13_gradcam.py` | Grad-CAM heatmap for a MobileNetV2 prediction |
| Confidence calibration | `14_calibrate_mobilenetv2.py` | Chooses a review threshold from validation predictions |
| Application | `15_app.py` | Streamlit UI, Grad-CAM, SQLite history and prediction analytics |
| Unified evaluation | `16_evaluate_all_models.py` | Compares only metrics carrying the same shared-split identifier |
| Split verification | `17_validate_shared_split.py` | Checks file presence, class coverage and cross-split duplicate hashes |

## Dataset setup

Put the original dataset in this structure:

```text
dataset/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
```

Folder names must be lowercase and match the six class names. The scripts support JPG, JPEG, PNG, BMP and WEBP.

The shared split generator:
- Uses a fixed random seed (42).
- Creates 70% train, 15% validation and 15% test partitions within each class.
- Computes SHA-256 hashes and skips exact duplicate images to reduce data leakage.
- Copies images to `shared_dataset/{train,val,test}/<class>/`.
- Writes `shared_dataset/manifest.csv` with source path, copied path, label, split and hash.
- Refuses to overwrite an existing split unless `--force` is explicitly supplied.

**Generate the split once and reuse it for every model.** Do not regenerate it between model runs, or the benchmark will no longer be the same.

## Environment

Use Python 3.10 or 3.11 in a virtual environment.

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow installation can differ by operating system. On Windows, use a supported TensorFlow/Python combination or WSL2 if your native Windows environment cannot install the required TensorFlow build. Install the compatible PyTorch/Ultralytics build for your hardware if needed.

## Complete training workflow

Run from the repository root, in this order.

### 1. Create the common split

```powershell
python 00_prepare_shared_dataset.py
python 17_validate_shared_split.py
```

If you intentionally want to recreate the split, first understand that all trained models and metrics will need to be regenerated. Then run:

```powershell
python 00_prepare_shared_dataset.py --force
```

### 2. Train traditional ML

```powershell
python 02_extract_shared_features.py
python 03_train_ml.py
```

The script trains Random Forest, KNN, Logistic Regression and SVM. It selects the saved best model using **validation macro-F1**, then evaluates each candidate on the same held-out test samples. It also stores class-wise reports and confusion matrices. PCA is included in the KNN/SVM feature pipelines to test dimensionality reduction against the full handcrafted feature representation.

For a single image:

```powershell
python 04_predict_ml.py .\\path\\to\\image.jpg
```

### 3. Train the custom CNN

```powershell
python 05_train_deep_learning.py
```

The CNN uses the same shared train, validation and test folders as the other model families.

### 4. Train MobileNetV2

```powershell
python 12_train_mobilenetv2.py
python 14_calibrate_mobilenetv2.py
```

Training occurs in two stages:
1. Freeze the ImageNet-pretrained MobileNetV2 backbone and train the classification head.
2. Unfreeze the upper 30 backbone layers and fine-tune them with a smaller learning rate. Batch-normalization layers remain frozen.

The confidence threshold is selected using validation predictions only. It is a manual-review aid, not a guarantee of correctness. The test set is not used to select the threshold.

### 5. Train and evaluate YOLOv8 classification

```powershell
python 09_train_yolo_classification.py
python 10_evaluate_yolo.py
```

This is YOLOv8 **classification**. It predicts one class for the full image; it does not locate multiple objects with bounding boxes.

### 6. Generate Grad-CAM

```powershell
python 13_gradcam.py .\\path\\to\\image.jpg
```

The heatmap is saved under `results/gradcam/`. Grad-CAM is an approximate visual explanation of regions influencing the model, not proof of causal reasoning.

### 7. Run the Streamlit application

```powershell
streamlit run 15_app.py
```

The app supports image upload, MobileNetV2 predictions, class probability display, optional Grad-CAM, a validation-calibrated manual-review warning, and saving predictions to `results/predictions.sqlite3`. The analytics section shows category counts and recent prediction history. GitHub Actions compiles Python files on pushes and pull requests; it does not install ML dependencies or run model training.

### 8. Run the webcam demo

After training YOLOv8:

```powershell
python 08_live_demo.py
```

The demo uses color contours only to propose a region of interest. It does not replace the model's predicted class or confidence. Press SPACE to count an item and Q to quit.

### 9. Generate the final model comparison

After all desired models have been trained and evaluated:

```powershell
python 16_evaluate_all_models.py
```

This script refuses to combine results unless each result file declares the same shared split ID and test evaluation protocol. Missing models are reported as not evaluated yet. It does not invent or fill missing metrics.

## Evaluation protocol

All model families should use:
- The same image-level train/validation/test manifest.
- Validation data for model selection, early stopping and confidence-threshold selection.
- The held-out test partition for final performance reporting only.
- Accuracy, macro precision, macro recall, macro F1, weighted F1, per-class precision/recall/F1 and a confusion matrix.

Macro F1 is especially useful when classes are imbalanced because it gives each class equal weight. The smaller Trash class should not disappear inside a weighted aggregate.

## Generated local artifacts

The following are generated locally and intentionally excluded from Git:
- `shared_dataset/`: copied images and manifest.
- `models/`: trained model weights and metadata.
- `results/`: metrics, plots, Grad-CAM images, calibration files and SQLite database.
- `runs/`: Ultralytics training outputs.

Do not commit private datasets, large model weights, local databases or generated training artifacts unless you have a specific reason and permission to distribute them.

## Current limitations and honest reporting

- `06_train_yolo.py` is retired and is not YOLO. Use the Ultralytics YOLOv8 classification scripts instead.
- MobileNetV2 transfer learning, Grad-CAM and the Streamlit/SQLite app are implemented as code, but must be run locally and checked with your actual dataset and installed package versions.
- A confidence score is not a calibrated probability unless calibration is empirically verified. The validation threshold is a review policy, not an accuracy guarantee.
- Grad-CAM is approximate and should be presented as a model-inspection aid.
- This image-level dataset does not support object-detection or multi-object localization claims.
- Do not quote accuracy numbers until the current scripts have been run and the saved results inspected.

## Team

| Name | GitHub | Contribution |
|---|---|---|
| Mehakdeep Kaur | [@Mehak-009deep](https://github.com/Mehak-009deep) | Coding, model testing, presentation and report |
| Krrish | [@e54trr5e](https://github.com/e54trr5e) | Coding and model training |
