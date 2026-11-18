# 🔶 Object Detection

> **Team:** Chati Pati | **Competition:** Quera Image Processing Olympiad

---

## 📖 Overview

The **Object Detection** module encompasses both state-of-the-art deep learning detectors (YOLO series) and deterministic classical computer vision pipelines. In competitive vision challenges, detection tasks range from high-level multi-object recognition in broadcast sports footage to exact sub-pixel geometric offside line analysis.

---

## 📂 Subdirectories & Projects

```
object_detection/
├── football_analysis/      # Deep learning: YOLOv8x player detection & K-Means jersey clustering
├── offside/                # Classical CV: Color-space VAR offside detection & geometric line fitting
├── sheet.ipynb             # Object Detection Bootcamp (IoU, NMS, YOLOv8, Roboflow, Synthetic data)
├── data/                   # Shared image data and Roboflow samples
├── runs/                   # Model training and inference outputs
└── yolov8n.pt              # YOLOv8 nano baseline weights
```

---

## 🏆 Featured Projects

### 1. ⚽ [Football Analysis](football_analysis/)
- **Core Technology:** `YOLOv8x` (68M parameters) + $K$-Means clustering in HSV space.
- **Workflow:** Detects players in broadcast football footage with low confidence thresholding ($0.20$) to capture distant players, removes false positives via bounding box area ratio filters ($0.0002 \le \text{area ratio} \le 0.05$), crops player regions, and clusters dominant jersey colors with $K$-Means ($k=2$) to classify teams automatically.
- **Documentation:** See [`football_analysis/README.md`](football_analysis/README.md).

### 2. 🚩 [Offside Detection](offside/)
- **Challenge:** [Quera #355584](https://quera.org/contest/assignments/103147/problems/355584)
- **Core Technology:** Pure-Python / NumPy deterministic color-space segmentation & geometric calibration (zero deep learning dependencies).
- **Workflow:** Standardizes image dimensions, detects on-screen legend palettes, creates multi-color masks, applies connected-component analysis and disc-template matching to isolate players (including white jerseys against pitch markings), computes the last defender line, and calculates the exact offside distance in meters using calibrated scale factors.
- **Output:** Returns `"Onside!"` or `"Offside!\n{distance:.2f}"`.
- **Documentation:** See [`offside/README.md`](offside/README.md).

---

## 📋 Curriculum: `sheet.ipynb`

Our structured laboratory notebook covers practical object detection workflows:
1. **Intersection over Union (IoU):** Vectorized bounding box overlap computation in $[x_1, y_1, x_2, y_2]$ coordinates.
2. **YOLOv8 Inference Pipeline:** Model instantiation, confidence thresholding, class filtering, and bounding box drawing.
3. **Robust Production Detection Wrapper:** Fail-safe inference pipeline with input validation, error handling, and structured dictionary output.
4. **Dataset Integration & Augmentation:** Parsing Roboflow / YOLO format annotation files (`labels/*.txt` with normalized $[c_x, c_y, w, h]$).
5. **Synthetic Data Generation:** Programmatic creation of colored geometric targets with ground-truth bounding box labels for unit testing.
6. **Model Fine-Tuning:** Training Ultralytics YOLO models on custom datasets with patience, early stopping, and metric tracking (mAP@50, mAP@50:95).

---

## 🚀 Quick Start

Run the offside detector:
```bash
python3 offside/offside_detection.py
```

Inspect the detection laboratory notebook:
```bash
jupyter notebook sheet.ipynb
```
