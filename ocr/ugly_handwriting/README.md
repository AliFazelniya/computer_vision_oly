# Ugly Handwriting — Optical Character Recognition

> **Competition:** Quera Image Processing Olympiad

---

## Project Overview

**Ugly Handwriting** is a challenging OCR (Optical Character Recognition) task where the goal is to recognize handwritten mathematical expressions and digits from low-quality, noisy images. The project employs a two-stage pipeline: first classifying the writing style, then recognizing individual characters using a custom CNN trained on a curated dataset of handwritten symbols including digits `0-9` and mathematical operators (`+`, `-`, `×`, `÷`, `(`, `)`).

---

## Problem Statement

Given a dataset of handwritten character images of varying quality, style, and noise level, build a system that:

1. **Classifies the writing type** (print vs. cursive) to apply appropriate preprocessing
2. **Recognizes the character** from 16 possible classes: `{0-9, +, -, ×, ÷, (, )}`

Key challenges:

- Extreme variability in stroke width and style
- Degraded/noisy images (blurring, smudging)
- Imbalanced class distribution
- Mixed mathematical symbols requiring precise segmentation

---

## Step-by-Step Solution

### Step 1: Preprocessing Pipeline

The `merge_vertical_boxes()` and `crop_and_resize_characters()` functions form the preprocessing backbone:

**`merge_vertical_boxes(boxes)`**:

- Sorts detected bounding boxes by X coordinate.
- Merges vertically adjacent boxes (operators written in multiple strokes) into a single region.
- Prevents over-segmentation of multi-stroke characters.

**`crop_and_resize_characters(binary_image, bounding_boxes)`**:

- Applies a configurable `padding=4` margin around each bounding box to avoid clipping character edges.
- Resizes each character crop to **28×28 pixels** — the standard input size for digit recognition CNNs (matching MNIST convention).
- Uses OpenCV for efficient batch cropping and resizing.

### Step 2: Type Classifier — Writing Style Recognition

```python
model = tf.keras.models.load_model('best_type_classifier.keras')
```

- A pre-trained Keras model predicts the **handwriting type** (print/cursive).
- This allows the downstream character recognizer to apply style-specific augmentation or preprocessing.
- `train_info.csv` provides type labels; class imbalance is handled with `compute_class_weight('balanced')`.

### Step 3: Custom OCR Dataset Construction

- **`TEMP_DIR` generation**: Reads the first 50 training images, extracts character regions using contour detection, and saves them to `temp_dataset/`.
- **`custom_ocr_dataset/`**: Organized into 16 class subdirectories (`0`, `1`, ..., `9`, `plus`, `minus`, `multiply`, `divide`, `l_paren`, `r_paren`).
- This curated dataset is used to train the final character recognizer from scratch.

### Step 4: Character Recognition Model — Custom CNN

```python
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
```

- **Architecture**: A sequential CNN with Conv2D → MaxPooling2D → Flatten → Dense layers (similar to LeNet-5).
- **Input**: 28×28 grayscale character images.
- **Output**: 16-class softmax over `{0-9, +, -, ×, ÷, (, )}`.
- **Augmentation**: Rotation ±10°, shifts, zoom to handle handwriting variability.
- **Class-weighted training**: `compute_class_weight('balanced')` compensates for rare operator symbols.

### Step 5: Inference & Submission

- Loads test images, applies the type classifier to determine preprocessing strategy.
- Runs character recognition CNN to predict class.
- Generates `submission.csv` with image filenames and predicted labels.

---

## Proposed Enhancements

| Enhancement                                          | Rationale                                                                                         |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **CRNN (CNN + LSTM)**                                | Handles sequences of characters without explicit segmentation — more robust for connected writing |
| **Attention-based Transformer OCR (TrOCR)**          | State-of-the-art handwriting recognition; handles arbitrary character sequences end-to-end        |
| **Data augmentation: elastic transforms**            | Simulates natural handwriting distortion better than affine augmentation                          |
| **Tesseract OCR + CNN ensemble**                     | Classic OCR engines provide strong priors for digit recognition                                   |
| **Contrastive pretraining (SimCLR)**                 | Learn stroke-invariant character representations from unlabeled handwriting images                |
| **CTC (Connectionist Temporal Classification) loss** | Enables word-level training without requiring character-level bounding boxes                      |
| **Higher resolution input (48×48)**                  | Reduces spatial aliasing in complex operator symbols like `×` and `÷`                             |
