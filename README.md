# AI-Based Smart Waste Segregation System

A machine learning project that classifies waste into 6 distinct categories: Cardboard, Glass, Metal, Paper, Plastic, and Trash, and triggers hardware servo actuation to route items into sorting bins.

## Project Structure
- `01_analyze_dataset.py`: Visualizes class distribution and dataset metrics.
- `02_extract_features.py`: Extracts HSV color histograms from dataset images.
- `03_train_ml.py`: Trains a Random Forest Classifier and evaluates metrics.
- `04_predict_ml.py`: Predicts waste category for an input image and simulates servo motor hardware output.

## Execution Steps
1. Place dataset images under `./dataset/<class_name>/`
2. Run `python 01_analyze_dataset.py`
3. Run `python 02_extract_features.py`
4. Run `python 03_train_ml.py`
5. Run `python 04_predict_ml.py ./path/to/test_image.jpg`