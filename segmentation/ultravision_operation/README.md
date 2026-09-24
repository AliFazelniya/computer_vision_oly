# Ultravision Operation — Medical X-Ray Segmentation & Classification

> **Competition:** Quera Image Processing Olympiad
> **Problem:** [Quera #310514](https://quera.org/contest/assignments/91552/problems/310514)

---

## Project Overview

**Ultravision Operation** is a dual-task medical imaging project that simultaneously performs **multi-class classification** (benign / malignant / normal) and **lesion segmentation** on chest X-ray images. The project uses a state-of-the-art two-headed architecture: a **CLIP ViT-B/32** vision-language model for classification and a **UNet-based segmentation model** (`segmentation_models_pytorch`) for pixel-level lesion delineation.

---

## Problem Statement

Given a dataset of chest X-ray images with three diagnostic categories:

- **Benign**: A non-cancerous growth or lesion visible in the X-ray
- **Malignant**: A cancerous lesion requiring urgent clinical attention
- **Normal**: No pathological finding present

Each training image is accompanied by a corresponding **segmentation mask** that outlines the region of interest (lesion). The model must:

1. Classify each test X-ray into one of the three categories
2. Predict the binary segmentation mask for the lesion region

**Key challenges:**

- Class imbalance (normal images dominate)
- Subtle intensity differences between benign and malignant lesions
- Precise mask prediction with thin and irregular lesion boundaries

---

## Step-by-Step Solution

### Step 1: Dataset Organization & Path Resolution

**`get_paths_and_label(row)`**:

- Reads the training CSV (`train.csv`) which contains image names and one-hot class columns (`class_benign`, `class_malignant`, `class_normal`).
- Resolves full image and mask paths from the structured `data/train/{label}/images/` and `data/train/{label}/masks/` directories.
- Returns a 3-tuple: `(img_path, mask_path, label_idx)`.

```
data/train/
├── benign/
│   ├── images/     ← X-ray image files
│   └── masks/      ← Corresponding binary lesion masks
├── malignant/
│   ├── images/
│   └── masks/
└── normal/
    ├── images/
    └── masks/
```

### Step 2: Dataset Class — `XRayDataset`

```python
img = cv2.imread(row['img_path'])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads as BGR; convert to RGB for PyTorch/albumentations
mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
mask = mask.astype(np.float32) / 255.0      # Normalize to [0, 1] binary mask
```

- **Missing masks**: Normal images have no lesion → mask defaults to an all-zero array.
- Images and masks pass through `albumentations` transforms together (synchronized augmentation).

### Step 3: Augmentation Pipeline (Albumentations)

| Transform                                     | Purpose                                                   |
| --------------------------------------------- | --------------------------------------------------------- |
| `A.Resize(224, 224)`                          | Standardize input dimensions for the backbone             |
| `A.HorizontalFlip(p=0.5)`                     | Mirror X-rays (lesions can appear on either side)         |
| `A.VerticalFlip(p=0.5)`                       | Further geometric diversity                               |
| `A.RandomRotate90(p=0.5)`                     | Handle varying image orientations                         |
| `A.Affine(scale, translate, rotate)`          | Simulate acquisition angle variability                    |
| `A.OneOf([ElasticTransform, GridDistortion])` | Simulate tissue deformation; critical for medical imaging |
| `A.RandomBrightnessContrast`                  | Simulate different X-ray exposure levels                  |
| `A.Normalize(ImageNet mean/std)`              | Standardize for pre-trained backbone compatibility        |
| `ToTensorV2()`                                | Convert numpy arrays to PyTorch tensors                   |

### Step 4: Classification Head — CLIP ViT-B/32

```python
MODEL_ID = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(MODEL_ID)
```

- **CLIP** (Contrastive Language-Image Pre-Training) is a vision-language model pre-trained on 400M image-text pairs.
- **Text prompts** define the three diagnostic classes in natural language:
  ```python
  text_prompts = [
      "an X-ray image of a benign tumor",
      "an X-ray image of a malignant tumor",
      "a normal X-ray image"
  ]
  ```
- **Fine-tuning strategy**: Only the **last encoder layer** and **`visual_projection`** are unfrozen — preserving learned representations while adapting to the X-ray domain.
- **`logits_per_image`**: The dot product between image embeddings and text embeddings gives a 3-class logit vector for classification.

### Step 5: Classification Training

```python
cls_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, clip_model.parameters()), lr=5e-5)
cls_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)   # Handles class imbalance
cls_scheduler = optim.lr_scheduler.ReduceLROnPlateau(mode='max', factor=0.5, patience=3)
cls_scaler = torch.amp.GradScaler('cuda')  # Automatic Mixed Precision (AMP) training
```

- **Balanced class weights** via `compute_class_weight('balanced')` — critical for the skewed normal/benign/malignant distribution.
- **AMP training** (`GradScaler`): Uses FP16 for forward pass, FP32 for weight updates — 2× speed improvement on modern GPUs.
- **ReduceLROnPlateau**: Halves LR when validation F1 plateaus for 3 epochs.

### Step 6: Segmentation Head — UNet (segmentation_models_pytorch)

```python
seg_model = smp.Unet(...)  # Encoder-decoder with skip connections
```

- **UNet architecture**: Skip connections between encoder and decoder paths preserve spatial detail for precise boundary prediction.
- Pre-trained encoder (e.g., ResNet34) transfers ImageNet features to the X-ray domain.
- Binary segmentation output with sigmoid activation and BCE+Dice combined loss.

### Step 7: Two-Stage Training & Submission

1. Train the CLIP classifier for 30 epochs
2. Train the segmentation model using the classified images as additional context
3. Save `best_classifier_model.pth` and `best_segmentation_model.pth`
4. Run inference on `data/test/images/`, generate `submission.csv` + segmentation mask images

---

## Proposed Enhancements

| Enhancement                                         | Rationale                                                                                                                             |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **MedCLIP or BioViL-T**                             | CLIP variants pre-trained on medical image-text pairs for better domain alignment                                                     |
| **Full CLIP fine-tuning with low LR**               | Adapting all layers with a very small LR (1e-6) may improve accuracy beyond last-layer fine-tuning                                    |
| **TransUNet / Swin-UNet**                           | Transformer-based segmentation models provide better global context for lesion boundary detection                                     |
| **Joint multi-task training**                       | Training classification and segmentation simultaneously with a shared encoder reduces computational cost and improves feature sharing |
| **Self-supervised pretraining on unlabeled X-rays** | MAE or SimCLR pretraining on the unlabeled test set before fine-tuning                                                                |
| **Monte Carlo Dropout for uncertainty**             | Quantifies prediction confidence — critical for clinical deployment                                                                   |
| **Connected component post-processing**             | Remove small isolated mask blobs with `cv2.connectedComponentsWithStats`                                                              |
| **TTA (Test-Time Augmentation)**                    | Average predictions over flipped/rotated versions of each test image                                                                  |
