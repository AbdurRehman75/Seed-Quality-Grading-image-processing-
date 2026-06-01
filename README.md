# SeedSense AI — Seed Quality Grading and Training Platform
University of Agriculture Faisalabad (UAF) · BS Software Engineering FYP 2026

**SeedSense** is an advanced seed quality grading application combining computer vision and deep learning to inspect, count, and classify grain seeds (wheat, rice, corn) into Grade A, B, or C.

## Overview: Hybrid AI Architecture

SeedSense uses a hybrid approach. OpenCV is used for seed detection, segmentation, counting, and visual feature extraction. A CNN model based on transfer learning is used for final Grade A/B/C classification when trained models are available. If CNN models are not present, the system safely falls back to OpenCV-based grading.

- **OpenCV Engine**: Performs contour detection, segmentation, seed counting, and computes individual metric scores:
  - *Shape Score* (circularity, aspect ratio, solidity)
  - *Color Score* (HSV pixel distribution deviation)
  - *Texture Score* (Laplacian edge variance)
  - *Broken/Foreign Ratio* (sizes and color thresholds)
- **CNN Classifier (Keras/MobileNetV2)**: Learns high-level textural, geometric, and chromatic patterns using ImageNet weights and deep learning fine-tuning.

### Realistic Accuracy & Design Guidelines
> [!NOTE]
> SeedSense achieves high-accuracy grading under controlled image conditions. Accuracy depends on dataset quality, image clarity, lighting, and seed separation.

> [!WARNING]
> Automated labeling is approximate. For scientific-grade accuracy, dataset labels should be reviewed by an agricultural expert.

---

## Installation & Setup

1. Clone or copy the project to your local directory.
2. Navigate to the project root:
   ```bash
   cd seedsense
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Directory Structure

```text
seedsense/
│
├── app.py                      # Main Flask application and SQLite database init
├── requirements.txt            # Project dependencies
├── database.db                 # SQLite database (auto-generated)
│
├── dataset/                    # Sorted and augmented training images (auto-generated)
│   ├── wheat/ [A, B, C]
│   ├── rice/  [A, B, C]
│   └── corn/  [A, B, C]
│
├── raw_datasets/               # Raw downloaded or generated seed images (auto-generated)
│   ├── wheat/
│   ├── rice/
│   └── corn/
│
├── models/                     # Trained Keras models and label lists (auto-generated)
│   ├── wheat_grade_model.keras
│   ├── wheat_labels.json
│   └── ...
│
├── training_reports/           # Confusion matrices and learning curves (auto-generated)
│   ├── wheat_metrics.json
│   ├── wheat_confusion_matrix.png
│   └── ...
│
├── services/                   # Backend Python ML pipeline services
│   ├── __init__.py
│   ├── dataset_builder.py      # Dataset downloader, quality grader, and augmenter
│   ├── train_model.py          # MobileNetV2 transfer learning and fine-tuner
│   ├── evaluate_model.py       # Metrics calculator and chart plotter
│   └── ml_predictor.py         # Thread-safe model cache and predictor
│
├── static/                     # Web uploads and processed output images
│   ├── uploads/
│   └── processed/
│
└── templates/                  # Frontend HTML/CSS templates
    ├── base.html               # Shared layout & nav
    ├── training.html           # AI Dashboard page
    └── ...
```

---

## Training Pipeline Commands

Follow these steps in your terminal (PowerShell / Command Prompt) to prepare the datasets and train the models.

### Step 1: Prepare and Augment the Dataset
This script downloads public Kaggle datasets (or automatically generates realistic mock seed images using OpenCV if Kaggle credentials are not configured). It filters out corrupt files, calculates OpenCV quality scores to auto-assign grades, and performs random augmentations (flips, rotations, contrast/brightness) to balance class sizes.

```bash
# Prepare all seed types and balance classes to 300 images each
python services/dataset_builder.py --seed_type all --min_per_class 300
```
*Options:*
- `--seed_type`: `wheat`, `rice`, `corn`, or `all`
- `--min_per_class`: Target count of images per grade (default: 300)

### Step 2: Train the CNN Classifier
Trains a customized MobileNetV2 network. It uses pre-trained ImageNet weights, trains classification heads, and optionally fine-tunes the last 20 layers. Integrates early stopping, checkpointing, and learning rate decay callbacks.

```bash
# Train models for all seed types for 25 epochs
python services/train_model.py --seed_type all --epochs 25
```
*Options:*
- `--seed_type`: `wheat`, `rice`, `corn`, or `all`
- `--epochs`: Total training cycles (default: 25)

### Step 3: Evaluate Models and Plot Metrics
Loads validation subsets, computes precision, recall, and f1-score metrics, and saves learning curves and confusion matrix charts to `training_reports/`.

```bash
# Evaluate models for all seed types
python services/evaluate_model.py --seed_type all
```

---

## Running the Web Server

Start the Flask application:
```bash
python app.py
```
Open your browser and navigate to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

1. Register or Log in.
2. Go to **AI Training** in the navbar to check the status of your models and view performance plots.
3. Go to **Analyze** to upload seed images and view hybrid OpenCV + CNN outputs.
4. Download quality reports as PDFs, view dashboards, or check grading history.
