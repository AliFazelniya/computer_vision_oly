# Watermark Detection — Image Authenticity Classification

> **Competition:** [Quera Image Processing Olympiad](https://quera.org/problemset/138169)

---

## Project Overview

**Watermark Detection** is a binary image classification task where the goal is to identify whether a given image contains a visible watermark or not. The project uses MobileNetV2 with a binary classification head, trained on a curated dataset of watermarked and clean images.

---

## Problem Statement

Given a dataset of images organized into `watermark` and `no_watermark` directories, train a binary classifier that can robustly detect the presence of any watermark (text overlay, logo, semi-transparent stamp) in a photograph.

**Key challenges:**

- Watermarks are visually diverse (text, logos, diagonal stamps)
- Watermarks may be semi-transparent or in corners (subtle)
- Must generalize to unseen watermark styles on test images

---

## Step-by-Step Solution

### Step 1: Dataset Discovery

- Scans `dataset/train/` directory using `os.listdir()` with extension filtering (`.png`, `.jpg`, `.jpeg`).
- Two subdirectories: `watermark/` and `no_watermark/` define the binary class labels.
- Constructs a test DataFrame from `dataset/test/` for inference.

### Step 2: Data Generator

```python
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)
```

- `rescale=1./255`: Normalizes pixels to `[0, 1]`.
- `flow_from_directory()`: Automatically infers labels from folder names.
- `validation_split=0.2`: Holds out 20% for validation during training.

### Step 3: Model Architecture — MobileNetV2 Transfer Learning

```
Input (128×128×3)
    └── MobileNetV2 (weights='imagenet', include_top=False) [Frozen]
        └── GlobalAveragePooling2D
            └── Dropout(0.2)
                └── Dense(1, activation='sigmoid')
```

- **`include_top=False`**: Removes the ImageNet classification head, exposing the 1280-dim feature map.
- **`GlobalAveragePooling2D()`**: Aggregates spatial features into a single 1280-dim vector — more compact and robust than Flatten.
- **`Dropout(0.2)`**: Regularization to prevent overfitting on the training set.
- **`sigmoid`** output: Direct probability of watermark presence.

### Step 4: Training

- **Loss:** Binary Cross-Entropy
- **Optimizer:** Adam (default LR = 1e-3)
- **Epochs:** Multiple with EarlyStopping (if configured)

### Step 5: Inference & Submission

- `flow_from_dataframe()` with `shuffle=False` for deterministic ordering.
- Predictions are thresholded at 0.5.
- Outputs `output.csv` with image names and binary predictions.

---

## Proposed Enhancements

| Enhancement                                               | Rationale                                                                                    |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Fine-tune top MobileNetV2 layers**                      | Adapt ImageNet features to watermark-specific textures                                       |
| **EfficientNetB4 at 224×224**                             | Higher resolution captures small corner watermarks missed at 128px                           |
| **Frequency-domain features (FFT)**                       | Watermarks often introduce regular frequency patterns — FFT magnitude maps can be diagnostic |
| **Attention visualization (GradCAM)**                     | Verify model focuses on watermark regions, not background artifacts                          |
| **Augmentation: random crop + brightness**                | Prevents spatial bias (watermarks usually in corners)                                        |
| **Focal Loss**                                            | Handles imbalanced watermark styles (subtle vs. bold)                                        |
| **Ensemble: CNN + Classical (Sobel + template matching)** | Classical methods can reliably detect text-based watermarks as a strong prior                |
