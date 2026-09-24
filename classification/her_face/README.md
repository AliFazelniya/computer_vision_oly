# Her Face — Multi-Attribute Face Classification

> **Competition:** Quera Image Processing Olympiad

---

## Project Overview

**Her Face** is a multi-label face attribute classification project. Given a photograph of a face, the model simultaneously predicts four binary attributes: **gender (Male)**, **age category (Young, Middle-Aged, Senior)**. The project uses transfer learning on top of MobileNetV2, fine-tuned with a multi-output classification head.

---

## Problem Statement

Given a dataset of face images with corresponding attribute labels (`Male`, `Young`, `Middle_Aged`, `Senior`), train a classifier that can predict all four attributes simultaneously from a single forward pass. Labels are provided in a CSV file with soft labels: `1` (positive), `-1` (negative), and `0` (uncertain), which are mapped to `1.0`, `0.0`, and `0.5` for the sigmoid-based multi-label training.

**Key challenges:**

- Class imbalance across attributes
- Label ambiguity (soft labels for uncertain cases)
- Generalization across lighting, pose, and age diversity

---

## Step-by-Step Solution

### Step 1: Data Preparation & Label Mapping

- Reads training metadata from `data/train.csv` and test metadata from `data/test.csv`.
- Maps soft labels: `{1 → 1.0, -1 → 0.0, 0 → 0.5}` to handle uncertain samples gracefully.
- Constructs image paths by combining `data/image_data/` with the `Filename` column.
- Splits into train/validation sets using `train_test_split`.

### Step 2: Data Generators (Keras `ImageDataGenerator`)

- `rescale=1./255` — normalizes pixel values to `[0, 1]`.
- Training augmentation: random horizontal flip, zoom, rotation for regularization.
- `flow_from_dataframe()` streams batches directly from disk to avoid memory issues.

### Step 3: Model Architecture — MobileNetV2 + Multi-output Head

```
Input (128×128×3)
    └── MobileNetV2 (weights='imagenet', include_top=False) [Frozen]
        └── GlobalAveragePooling2D
            └── Dropout(0.2)
                └── Dense(4, activation='sigmoid')   ← One output per attribute
```

- **MobileNetV2** pre-trained on ImageNet provides robust low-level and mid-level feature extraction.
- `base_model.trainable = False` — backbone is frozen to prevent catastrophic forgetting on the small face dataset.
- `sigmoid` output: each of the 4 outputs is independently thresholded at 0.5 for binary decision.

### Step 4: Training

- **Loss:** Binary Cross-Entropy (one per output, summed)
- **Optimizer:** Adam with default LR
- **Metrics:** Accuracy per attribute, F1 score

### Step 5: Evaluation & Submission

- Runs inference on the test generator.
- Applies 0.5 threshold to sigmoid outputs.
- Generates a submission CSV with predicted attribute values.

---

## Proposed Enhancements

| Enhancement                                  | Rationale                                                                                            |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Unfreeze top MobileNetV2 layers**          | Fine-tuning the upper layers on face data significantly improves attribute-specific feature learning |
| **EfficientNetV2-S or ViT-B/16**             | Larger capacity backbones with better ImageNet initialization for fine-grained attributes            |
| **Weighted Binary Cross-Entropy**            | Down-weight uncertain (`0.5`) samples; up-weight rare positive cases                                 |
| **Focal Loss**                               | Addresses class imbalance by focusing training on hard examples                                      |
| **Augmentation: FaceAug / albumentations**   | Face-specific augmentations (occlusion, skin-tone jitter) improve robustness                         |
| **Multi-task learning with auxiliary heads** | Adding an age regression head alongside classification forces richer age representations             |
| **Label smoothing**                          | Treats soft labels more rigorously; prevents over-confident predictions                              |
