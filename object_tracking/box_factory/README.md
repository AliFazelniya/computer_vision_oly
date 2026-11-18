# 📦 Box Factory — Industrial Quality Control & Defective Box Tracking

> **Competition:** Quera Image Processing Olympiad  
> **Problem Link:** [Quera #310514 / Assignment #109270](https://quera.org/contest/assignments/109270/problems)

---

## 📖 Project Overview

In automated manufacturing lines, quality control vision systems must rapidly inspect conveyor belts, distinguish intact products from defective items, and track individual units across video frames to prevent duplicate counting.

**Box Factory** tackles automated package quality inspection using **YOLO11s** object detection paired with continuous video tracking (`CenterTracker`). The system classifies packages into two distinct categories:
- **`healthy_box` (Class 0):** Undamaged, correctly folded and sealed packages.
- **`defective_box` (Class 1):** Crushed, torn, misaligned, or unsealed packages.

---

## 🛠️ End-to-End Pipeline & Architecture

```
Conveyor Video Stream
          │
          ▼
Frame Extraction & Preprocessing (640x640)
          │
          ▼
YOLO11s Fine-Tuned Detector ──> Bounding Boxes & Confidence Scores
          │
          ▼
CenterTracker / Spatial Matching ──> Inter-frame Object ID Association
          │
          ▼
Temporal Trajectory Filtering ──> Reject Transient Ghost Detections
          │
          ▼
Defective vs. Healthy Counting ──> Generate submission.csv
```

### 1. Dataset Preparation & Structuring
- Automated Google Drive ingestion via `gdown`.
- Verification of image formats (`.jpg`, `.png`, `.bmp`, `.webp`) and label parsing.
- Dataset partitioning into standard YOLO directory hierarchy (`images/train`, `images/val`, `labels/train`, `labels/val`) defined in `box_factory_data.yaml`.

### 2. Model Training Configuration (`yolo11s.pt`)
- **Backbone Architecture:** Ultralytics YOLO11s (small variant, ~9.4M parameters), providing optimal precision-to-latency ratio for real-time edge deployment.
- **Image Size:** $640 \times 640$ pixels.
- **Training Epochs:** 100 with Early Stopping (`patience=20`).
- **Batch Size:** 4 (adaptive based on CUDA availability).
- **Optimizer:** Auto-tuned AdamW with weight decay and cosine learning rate scheduling.

### 3. Video Tracking Engine (`CenterTracker`)
- Computes Euclidean distance between the centroids of existing tracked boxes and incoming frame detections.
- Associates detections to active track IDs using minimum-distance assignment within a maximum distance threshold.
- Handles track registration for newly entering boxes and deregistration for boxes exiting the conveyor boundary.
- Employs temporal persistence buffers to maintain track IDs through temporary occlusions or lighting glares.

### 4. Evaluation & Submission Generation
- Processes test conveyor videos from `test/videos/raw/`.
- Aggregates defect counts per video sequence into `submission.csv` according to the required competition schema.

---

## 📂 File Directory

| File / Folder | Purpose |
| :--- | :--- |
| **`main.ipynb`** | Complete pipeline notebook (download, dataset prep, YOLO11s training, tracking, inference) |
| **`box_factory_data.yaml`** | YOLO dataset configuration file specifying train/val paths and class names |
| **`box-factory-dataset/`** | Dataset directory containing train/val images and annotations |
| **`submission.csv`** | Final competition predictions |
| **`yolo11n.pt`, `yolo11s.pt`** | Base and fine-tuned model checkpoints |

---

## 🚀 Quick Start

Open and execute the complete pipeline in Jupyter:
```bash
jupyter notebook main.ipynb
```
