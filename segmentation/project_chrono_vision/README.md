# Project Chrono Vision — Instance Segmentation

> **Competition:** Quera Image Processing Olympiad

---

## Project Overview

**Project Chrono Vision** is an instance segmentation project using **YOLOv8l-seg** (the large segmentation variant) to perform pixel-level object delineation across complex, multi-object scenes. The project processes images through the YOLOv8 segmentation pipeline and outputs per-instance segmentation masks along with class labels and confidence scores.

---

## Problem Statement

Given a dataset of competition images containing multiple object instances, produce:

1. **Bounding boxes** for each detected object instance
2. **Pixel-accurate segmentation masks** for each instance
3. **Class labels and confidence scores** for each detection

**Key challenges:**

- Instance separation: objects of the same class overlapping
- Handling small, partially visible instances
- Efficient mask generation at competition latency requirements

---

## Step-by-Step Solution

### Step 1: Model Selection — YOLOv8l-seg

```
yolov8l-seg.pt  ← Large segmentation variant (~43M parameters)
```

- **YOLOv8-seg** extends the standard YOLOv8 detection head with a **mask prediction branch**.
- The mask branch uses a prototype-based approach: generates K prototype masks, then predicts per-instance linear combination coefficients.
- The `l` (large) variant balances accuracy and inference speed, appropriate for competition submissions.

### Step 2: Dataset Preparation

- Images are organized in `data/` with corresponding annotation files.
- Submission structure uses named subdirectories (`I_BS_01`, `I_CA_01`, etc.) where the prefix encodes image set identifiers.

### Step 3: Inference Pipeline

```python
model = YOLO('yolov8l-seg.pt')
results = model(image_path)
# Each result contains:
#   result.boxes   ← Bounding boxes [x1, y1, x2, y2, conf, cls]
#   result.masks   ← Binary segmentation masks per instance
```

### Step 4: Mask Post-Processing

- Masks are upsampled back to the original image resolution.
- Per-instance masks are stored in the `submission/` directory with the competition-expected naming convention.

### Step 5: Submission Generation

- For each test image, writes the segmentation masks to the appropriate `submission/` subdirectory.
- File naming follows the competition schema for automated evaluation.

---

## Proposed Enhancements

| Enhancement                       | Rationale                                                                        |
| --------------------------------- | -------------------------------------------------------------------------------- |
| **YOLOv8x-seg fine-tuning**       | Training on domain-specific data dramatically improves mask quality              |
| **Mask R-CNN (Detectron2)**       | Provides higher-quality masks through RoI Align + separate mask head             |
| **SAM 2 (Segment Anything v2)**   | Zero-shot instance segmentation with superior boundary precision                 |
| **CRF post-processing**           | Dense CRF refines mask boundaries using image color statistics                   |
| **Multi-scale inference (TTA)**   | Test-time augmentation at multiple scales improves small instance recall         |
| **NMS tuning for dense scenes**   | Lowering IoU threshold for NMS prevents aggressive merging of adjacent instances |
| **Contour-based mask refinement** | Apply `cv2.findContours` + `cv2.convexHull` to smooth jagged YOLO mask edges     |
