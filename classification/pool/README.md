# Pool Detection — Property Listing Image Classification

> **Competition:** [Quera Image Processing Olympiad](https://quera.org/problemset/251285)

---

## Project Overview

**Pool Detection** (`EstakhtemoonNasheh`) is a binary image classification project targeting real estate listing images. The goal is to determine whether a property image shows a **swimming pool** or not. The project parses JSON-structured listing data, extracts image metadata, and trains a MobileNetV2-based binary classifier.

---

## Problem Statement

Given a dataset of real estate listing images with associated metadata (room title, description, amenities) provided in JSON format, build a model that classifies each image as:

- **1 (Pool present)**
- **0 (No pool)**

The dataset has significant class imbalance (pools are rare) and high visual diversity (indoor shots, aerial views, exterior shots).

---

## Step-by-Step Solution

### Step 1: Data Ingestion from JSON

- Reads structured listing data from `data/train.json` and `data/test.json` using `pd.read_json()`.
- Uses `pd.json_normalize()` to flatten nested `rooms` objects into a flat DataFrame.
- **`extract_pool_label()`**: Scans the `title` and `description` fields for pool-related keywords to generate ground-truth binary labels.
- Constructs paths to images within `data/train/` and `data/test/` directories.

### Step 2: Data Generator Setup

- Uses Keras `ImageDataGenerator` with:
  - `rescale=1./255` — pixel normalization to `[0, 1]`
  - `validation_split=0.2` — 80/20 train-validation split
- `flow_from_dataframe()` with `class_mode='binary'` for sigmoid-based binary output.
- Target size: **128×128** (trade-off between detail and compute).

### Step 3: Model Architecture — MobileNetV2 + Binary Head

```
Input (128×128×3)
    └── MobileNetV2 (weights='imagenet', include_top=False) [Frozen]
        └── GlobalAveragePooling2D
            └── Dropout(0.2)
                └── Dense(1, activation='sigmoid')   ← Pool / No Pool
```

### Step 4: Training

- **Loss:** Binary Cross-Entropy
- **Optimizer:** Adam
- Threshold at 0.5 for final prediction

### Step 5: Evaluation & Submission

- `model.predict(val_generator)` → binarized at 0.5
- Reports F1 score, classification report, and confusion matrix
- Generates submission CSV

---

## Proposed Enhancements

| Enhancement                           | Rationale                                                                                        |
| ------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Class-weighted loss**               | Addresses the severe imbalance (pools are rare in listings)                                      |
| **Text + Image fusion**               | Concatenate TF-IDF or BERT embeddings of the listing text with image features for a richer model |
| **Higher resolution (224×224)**       | Pools in aerial shots may be missed at 128×128                                                   |
| **EfficientNetB3 or ResNet50**        | Better capacity than MobileNetV2 for fine-grained visual detection                               |
| **Contrastive Data Augmentation**     | Emphasize pool-specific textures (blue water, tile patterns)                                     |
| **Threshold tuning**                  | Optimize F1 by adjusting the decision threshold away from 0.5 using the validation set           |
| **Ensemble with XGBoost on metadata** | Combine textual features with CNN predictions for a boosted final decision                       |
