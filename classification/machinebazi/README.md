# Machinebazi — Multi-Label Car Part Classification

> **Competition:** Quera Image Processing Olympiad
> **Problem:** [Quera #307569](https://quera.org/problemset/307569)

---

## Project Overview

**Machinebazi** is a multi-label image classification task where the goal is to identify which car parts are present in a given 640×640 photograph. The project fine-tunes a **ResNet50** backbone (trained from scratch — no pretrained weights) with a multi-class head, using class-weighted loss to handle the inherent label imbalance across different car part categories.

---

## Problem Statement

Given a training dataset of 640×640 car part images with one-hot encoded class labels (provided in `train/train.csv`), train a classifier that:

1. Correctly identifies the car part category from an image
2. Generalizes to unseen test images (`test/test.csv`)

The task has multiple classes representing distinct car components. Class imbalance is a key challenge: some parts appear far more frequently than others in the dataset.

---

## Step-by-Step Solution

### Step 1: Data Ingestion & Preprocessing

```python
train_df = pd.read_csv("train/train.csv")
test_df  = pd.read_csv("test/test.csv")
```

- All class columns are explicitly cast to `int` to prevent dtype inference errors.
- Filenames are stripped of whitespace for robust path construction.
- `class_cols`: all columns except `filename` — these are the one-hot label columns.

### Step 2: Stratified Train/Validation Split

```python
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
```

- **StratifiedShuffleSplit** ensures the 15% validation set has the same class distribution as the full training set — critical for reliable F1 evaluation when classes are imbalanced.
- The dominant class label (`argmax` of the one-hot row) is used as the stratification key.

### Step 3: Class Weight Computation

```python
class_weights = 1.0 / np.maximum(class_counts, 1)
class_weights = class_weights / class_weights.sum() * len(class_cols)
```

- Inverse-frequency weighting: rare classes get higher weights.
- Normalized to sum to `num_classes` to keep the absolute loss scale stable.
- Passed to `nn.CrossEntropyLoss(weight=...)` so the model penalizes errors on rare classes more heavily.

### Step 4: Data Augmentation Pipeline

```python
train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

- **RandomHorizontalFlip**: Car parts appear symmetrically in both orientations.
- **RandomRotation(10°)**: Handles slight camera angles without destroying part identity.
- **ColorJitter**: Simulates different lighting conditions and camera white-balance settings.
- **ImageNet Normalization**: Standardizes to the same distribution as ImageNet — important even when not using pretrained weights, as it centers the data for stable training.

### Step 5: CarPartDataset (PyTorch Dataset)

```python
y = int(np.argmax(row[class_cols].values))
```

- Converts one-hot labels to a single integer class index via `argmax`.
- Opens images with `PIL.Image.open().convert("RGB")` and applies the transform pipeline.
- Returns `(image_tensor, class_index)` tuples for the DataLoader.

### Step 6: Model Architecture — ResNet50

```python
m = models.resnet50(weights=None)   # Trained from scratch
m.fc = nn.Linear(in_feats, num_classes)
```

- **ResNet50**: 4 residual stages (3-4-6-3 blocks), ~23M parameters. Residual connections prevent vanishing gradients in deep networks.
- `weights=None` means all weights are randomly initialized — the model learns entirely from the competition training data.
- The final `fc` (fully connected) layer is replaced with one matching the number of car part classes.

Other supported backbones (commented out for easy switching):

- `resnet18`, `resnet34` — faster, lighter
- `efficientnet_b0` — compound scaling (width/depth/resolution)
- `convnext_tiny` — modern CNN with transformer-inspired design
- `vit_b_16` — Vision Transformer at 224×224

### Step 7: Training Loop

```python
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
```

- **AdamW**: Adam with decoupled weight decay — better regularization than standard L2.
- **CrossEntropyLoss with class weights**: Focuses gradient updates on under-represented classes.
- 50 epochs with per-epoch F1 score evaluation on the validation set.
- Best model (highest F1) is saved and used for test inference.

### Step 8: Inference & Submission

- Test images pass through `eval_tfms` (Resize + Normalize, no augmentation).
- `argmax` of softmax probabilities gives the predicted class.
- Results written to `submission.csv`.

---

## Proposed Enhancements

| Enhancement                      | Rationale                                                                                     |
| -------------------------------- | --------------------------------------------------------------------------------------------- |
| **ImageNet pretrained weights**  | Even `weights='DEFAULT'` from PyTorch Hub can give a +5-10% F1 boost vs training from scratch |
| **Label smoothing**              | `CrossEntropyLoss(label_smoothing=0.1)` prevents overconfident predictions                    |
| **Cosine Annealing LR**          | Smooth LR decay from 3e-4 → 0 often outperforms constant LR for 50 epochs                     |
| **Mixup / CutMix augmentation**  | Interpolates pairs of images in feature space — powerful regularizer for small datasets       |
| **EfficientNetV2-S**             | Better accuracy/parameter tradeoff than ResNet50 at the same image size                       |
| **Test-Time Augmentation (TTA)** | Average predictions over multiple augmented versions of each test image                       |
| **Focal Loss**                   | Automatically up-weights hard-to-classify examples regardless of class frequency              |
| **k-fold cross-validation**      | More reliable evaluation and ensembling: train 5 models on different folds                    |
