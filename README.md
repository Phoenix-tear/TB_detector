# Tuberculosis (TB) Detection from Chest X-ray Images

This project implements a deep learning model to classify chest X-ray images as either **"Normal"** or **"Tuberculosis"**. It utilizes **transfer learning** with the **VGG16** architecture and is built using **TensorFlow** and **Keras**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tOeiFVxFzkuizlvberFLGh0jS6-2vILf?usp=sharing)

---

## 📌 Project Overview

The primary goal is to build an accurate classifier for TB detection, which can be a valuable tool in medical imaging.  
This notebook covers the entire pipeline:
- Data acquisition and preprocessing
- Model training and fine-tuning
- Evaluation and saving

---

## 📁 Dataset

- **Source**: [tawsifurrahman/tuberculosis-tb-chest-xray-dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
- **Classes**:
  - `Normal`: Chest X-rays without TB.
  - `Tuberculosis`: Chest X-rays indicating TB.

The dataset is downloaded using **kagglehub** and copied to local Colab storage for faster access during training.

---

## ⚙️ Workflow

### 🔹 Dataset Preparation
- Download dataset from Kaggle.
- Organize it under `TB_Chest_Radiography_Database/`.
- Copy to `/content/fast_dataset` for efficient training.

### 🔹 Data Augmentation & Preprocessing
- Use `ImageDataGenerator` for:
  - Rescaling
  - Rotation
  - Shifts
  - Shear
  - Zoom
  - Horizontal flip
- Resize images to `(224, 224)`.

### 🔹 Model Building (Transfer Learning)
- **Base Model**: VGG16 (pre-trained on ImageNet, `include_top=False`)
- **Frozen layers**: All base layers frozen initially.
- **Custom head**:
  - `GlobalAveragePooling2D`
  - `Dense(512, activation='relu')`
  - `BatchNormalization`
  - `Dropout(0.5)`
  - `Dense(2, activation='softmax')` (for binary classification)

### 🔹 Model Compilation
- **Optimizer**: `Adam`
- **Loss**: `categorical_crossentropy`
- **Metrics**: `accuracy`

### 🔹 Callbacks
- `EarlyStopping`: Prevents overfitting by monitoring validation loss.
- `ReduceLROnPlateau`: Reduces learning rate when validation loss plateaus.

### 🔹 Initial Training
- Train only the custom head (VGG16 layers frozen).

### 🔹 Fine-Tuning
- Unfreeze VGG16 layers from layer 15 onward.
- Re-compile with a lower learning rate.
- Continue training.

### 🔹 Evaluation
- Plot training/validation accuracy and loss.
- Generate:
  - Classification report (Precision, Recall, F1-score)
  - Confusion matrix
  - AUC-ROC score

### 🔹 Model Saving
- Save model as:
  - `lung_disease_classifier.h5`
  - `lung_disease_classifier.keras`

---

## 📦 Requirements

The following Python libraries are required:

- `kagglehub`
- `tensorflow >= 2.0`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `numpy`
- `pandas`

---

## 🧠 Model Architecture

- **Base**: VGG16 (ImageNet weights, no top layers)
- **Custom Head**:
  - `GlobalAveragePooling2D`
  - `Dense(512, activation='relu')`
  - `BatchNormalization`
  - `Dropout(0.5)`
  - `Dense(2, activation='softmax')`

---

## 📊 Results

The notebook aims for **high accuracy** in classifying TB vs. Normal X-rays. Example final metrics:

- **Final Accuracy**: ~0.9929
- **Final F1-Score**: ~0.9923
- **AUC-ROC Score**: ~0.9993

> These metrics may vary with dataset updates or training randomness.

---

## 🚀 How to Use

1. **Open in Google Colab**: Click the badge at the top.
2. **Setup Kaggle API** (if required): Upload your `kaggle.json` for `kagglehub` access.
3. **Run All Cells**: Execute notebook cells sequentially.
   - **Tip**: Enable GPU (Runtime > Change runtime type > GPU).
4. **View Results**: Training logs, metrics, and plots will be displayed.
5. **Model Access**: Saved models (`.h5`, `.keras`) are available for download or saving to Google Drive.

---

## 🧪 Potential Improvements

- Try different pre-trained architectures (e.g., ResNet, Inception, EfficientNet).
- Use more advanced augmentation techniques.
- Perform hyperparameter tuning.
- Handle class imbalance (if present) with techniques like:
  - Weighted loss
  - SMOTE
  - Undersampling/oversampling

---
