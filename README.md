<div align="center">

# Team Chati Pati — Quera Image Processing Olympiad

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)](https://opencv.org)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v8%20%7C%2011-yellow)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A comprehensive, production-grade repository of Computer Vision solutions, algorithmic pipelines, and curriculum notebooks developed by Team "Chati Pati" for the Quera Image Processing Olympiad — Final Round.**

</div>

<p align="center"><img src="poster.jpg" alt="Team Chati Pati Poster" width="90%" /></p>

---

## 📖 About This Repository

This repository serves as the unified engineering hub for **Team Chati Pati**'s solutions across all phases of the [Quera Image Processing Olympiad](https://quera.org). It spans the entire modern computer vision spectrum: from deterministic signal processing and frequency-domain filtering to deep convolutional networks, vision-language models (CLIP), real-time object detection/tracking (YOLOv8/11 + SORT), and complex Persian OCR systems.

The repository is organized into dedicated discipline directories. Each project contains its own self-contained code, detailed technical README, pipeline architectures, hyperparameter configs, and evaluation metrics.

---

## 🗂️ Complete Directory Hierarchy

```
computer_vision_oly/
│
├── classification/                     # Image Classification & Multi-Attribute Learning
│   ├── her_face/                       # Multi-attribute facial classification (MobileNetV2)
│   ├── machinebazi/                    # Multi-label car component classification (ResNet50 from scratch)
│   ├── pool/                           # Real estate swimming pool classification from multimodal listings
│   └── watermark_detection/            # Binary image watermark presence detection
│
├── object_detection/                   # Object Detection, Bounding Box Regression & VAR Analytics
│   ├── football_analysis/              # YOLOv8x player detection & K-Means jersey color clustering
│   ├── offside/                        # Classical VAR offside line detection via color-space segmentation
│   └── sheet.ipynb                     # Phase 2: Object Detection Bootcamp (IoU, NMS, YOLOv8, Roboflow)
│
├── object_tracking/                    # Video Multi-Object Tracking & Motion Estimation
│   ├── box_factory/                    # Industrial conveyor defect tracking (YOLO11s + CenterTracker)
│   ├── sheet.ipynb                     # Phase 3: Tracking Master Laboratory (Centroid, Hungarian, SORT, ByteTrack)
│   └── papers/                         # Research literature and algorithmic citations
│
├── ocr/                                # Optical Character Recognition & RTL Script Engineering
│   ├── packman/                        # Maze perspective rectification, entity detection & scoreboard OCR
│   ├── ugly_handwriting/               # Degraded handwritten math expression OCR (16-class CNN)
│   ├── phase4_persian_ocr/             # Persian OCR benchmark research: calibration, multiline, synthetic data
│   └── sheet.ipynb                     # Phase 4: Master Persian OCR Notebook (Unicode, Raqm, Tesseract, CER)
│
├── segmentation/                       # Pixel-Level Delineation & Audio-Visual Segmentation
│   ├── avengers_shield/                # 1D temporal drone audio event segmentation (Dilated TCN)
│   ├── project_chrono_vision/          # Instance segmentation using YOLOv8l-seg
│   └── ultravision_operation/          # Medical chest X-ray lesion segmentation (CLIP ViT + UNet)
│
├── geometry_and_puzzle_solving/        # Structured Geometric Rectification & Constraint Solvers
│   └── sudoku/                         # Photographic Sudoku extraction + SudokuCNN + MRV backtracking solver
│
├── image_processing_fundamentals/      # Classical Computer Vision & Signal Processing
│   ├── postcard/                       # 2D FFT periodic noise removal & distance-transform glow watermarking
│   ├── skin_finding/                   # Dual color-space (YCrCb + HSV) face skin segmentation with IoU eval
│   └── sheet.ipynb                     # Phase 1: 7-Day OpenCV Professional Mastery Bootcamp
│
├── integrated_problems/                # Capstone Multi-Domain Vision Pipelines & Contest Prep
│   ├── notebook.ipynb                  # Phase 5: Master Integration Notebook (Wall geometry, MOT, OCR, Sweeps)
│   ├── competition-prep/               # Modular, self-testing CV toolbox (geometry, tracking, OCR, evaluation)
│   └── integrated_problems/            # Benchmark datasets, synthetic crossing videos, and stress reports
│
├── docs/                               # Team Documentation, References & Presentation Slides
│   ├── OCR_Project_Study_Note.pdf      # In-depth architectural notes on end-to-end OCR pipelines
│   ├── yolo.pdf                        # Theoretical evolution of YOLO detectors
│   └── yolo.pptx                       # Team seminar presentation slides on modern real-time detection
│
├── poster.jpg                          # Team Chati Pati visual poster
├── requirements.txt                    # Project-wide Python dependencies
└── README.md                           # Master landing documentation
```

---

## 🏆 Projects at a Glance

| Category | Project | Core Techniques & Models | Challenge / Link |
| :--- | :--- | :--- | :--- |
| **Classification** | [`machinebazi`](classification/machinebazi/) | ResNet50 trained from scratch, class-weighted CE, StratifiedSplit | [Quera #307569](https://quera.org/problemset/307569) |
| **Classification** | [`her_face`](classification/her_face/) | MobileNetV2 transfer learning, multi-output sigmoid heads | [Quera #251284](https://quera.org/problemset/251284) |
| **Classification** | [`pool`](classification/pool/) | JSON metadata parsing + MobileNetV2 image classification | [Quera #251285](https://quera.org/problemset/251285) |
| **Classification** | [`watermark_detection`](classification/watermark_detection/) | MobileNetV2 binary classification, data generator streaming | [Quera #138169](https://quera.org/problemset/138169) |
| **Object Detection** | [`offside`](object_detection/offside/) | Pure-Python color segmentation, disc edge finding, scale calibration | [Quera #355584](https://quera.org/contest/assignments/103147/problems/355584) |
| **Object Detection** | [`football_analysis`](object_detection/football_analysis/) | YOLOv8x player detection, area ratio filters, $K$-Means jersey clustering | [Quera #307571](https://quera.org/problemset/307571) |
| **Object Detection** | [`sheet.ipynb`](object_detection/sheet.ipynb) | IoU vectorized calculation, YOLOv8 nano, Roboflow parsing, synthetic targets | Phase 2 Curriculum |
| **Object Tracking** | [`box_factory`](object_tracking/box_factory/) | YOLO11s package defect detection + CenterTracker conveyor tracking | [Quera #310514](https://quera.org/contest/assignments/109270/problems) |
| **Object Tracking** | [`sheet.ipynb`](object_tracking/sheet.ipynb) | Centroid tracker, Hungarian bipartite matching, Kalman filtering, SORT, ByteTrack | Phase 3 Curriculum |
| **OCR** | [`packman`](ocr/packman/) | Maze quad perspective warping, entity masks, PaddleOCR/EasyOCR scoreboard read, BFS | [Quera #363305](https://quera.org/contest/assignments/109270/problems/363305) |
| **OCR** | [`ugly_handwriting`](ocr/ugly_handwriting/) | Multi-stroke box merging, style classification, 16-class LeNet CNN | [Quera #292198](https://quera.org/problemset/292198) |
| **OCR** | [`phase4_persian_ocr`](ocr/phase4_persian_ocr/) | Persian OCR benchmarking, multiline transcriptions, degradation calibration | Research Benchmark |
| **OCR** | [`sheet.ipynb`](ocr/sheet.ipynb) | RTL Unicode normalisation, Raqm font rendering, contrast stretching, Tesseract CER | Phase 4 Curriculum |
| **Segmentation** | [`avengers_shield`](segmentation/avengers_shield/) | 1D Dilated Temporal Convolutional Network (TCN) + BoundaryRefiner | [Quera #355588](https://quera.org/contest/assignments/103147/problems/355588) |
| **Segmentation** | [`ultravision_operation`](segmentation/ultravision_operation/) | Dual-headed CLIP ViT-B/32 classifier + `smp` UNet lesion segmentation | [Quera #310514](https://quera.org/contest/assignments/91552/problems/310514) |
| **Segmentation** | [`project_chrono_vision`](segmentation/project_chrono_vision/) | YOLOv8l-seg instance segmentation, prototype mask recombination | [Quera #310516](https://quera.org/contest/assignments/91552/problems/310516) |
| **Geometry** | [`sudoku`](geometry_and_puzzle_solving/sudoku/) | Adaptive threshold warp, 9x9 cell slicer, SudokuCNN, MRV backtracking | [Quera #355590](https://quera.org/contest/assignments/103147/problems/355590) |
| **Fundamentals** | [`postcard`](image_processing_fundamentals/postcard/) | 2D FFT notch filter, distance-transform glow/shadow, parallel SSIM grid search | [Quera #307570](https://quera.org/problemset/307570) |
| **Fundamentals** | [`skin_finding`](image_processing_fundamentals/skin_finding/) | Dual color-space ($YCrCb \land HSV$), elliptical morphology, hole filling, IoU | [Quera #292197](https://quera.org/problemset/292197) |
| **Fundamentals** | [`sheet.ipynb`](image_processing_fundamentals/sheet.ipynb) | 7-day OpenCV mastery: Sobel, Canny, Hough lines/circles, ORB+RANSAC, TLS | Phase 1 Curriculum |
| **Integration** | [`integrated_problems`](integrated_problems/) | Capstone Phase 5: Wall edge TLS regression, MOT crossing counters, 60-min mock harness | Phase 5 Integration |

---

## 🛠️ Setup & Installation

### Prerequisites
- Linux / macOS / Windows with **Python 3.10+**
- NVIDIA GPU with CUDA 12.1+ support (recommended for deep learning modules)
- Git & Git LFS

### 1. Clone the Repository
```bash
git clone https://github.com/AliFazelniya/computer_vision_oly.git
cd computer_vision_oly
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*Key libraries included:* `torch`, `torchvision`, `ultralytics`, `opencv-python-headless`, `scipy`, `scikit-learn`, `scikit-image`, `tensorflow`, `transformers`, `filterpy`, `pillow`, `tqdm`.

---

## 🚀 Execution & Quick Start

### Run Standalone Python Pipelines
```bash
# Classical Offside Detection
python3 object_detection/offside/offside_detection.py

# Dual Color-Space Skin Segmentation
python3 image_processing_fundamentals/skin_finding/skin.py

# FFT Postcard Denoising & Distance-Transform Glow Watermark
python3 image_processing_fundamentals/postcard/watermark_v2.py

# Parallel SSIM Hyperparameter Optimization
python3 image_processing_fundamentals/postcard/grid_search.py

# Run Integrated Competition-Prep Self-Tests
python3 integrated_problems/competition-prep/run_all_tests.py
```

### Launch Interactive Jupyter Notebooks
```bash
# Phase 1: 7-Day OpenCV Professional Mastery
jupyter notebook image_processing_fundamentals/sheet.ipynb

# Phase 2: Object Detection Mastery
jupyter notebook object_detection/sheet.ipynb

# Phase 3: Multi-Object Tracking Mastery
jupyter notebook object_tracking/sheet.ipynb

# Phase 4: Persian OCR Mastery
jupyter notebook ocr/sheet.ipynb

# Phase 5: Multi-Domain Integration & Mock Contest
jupyter notebook integrated_problems/notebook.ipynb

# Deep Learning Challenges
jupyter notebook classification/machinebazi/notebook.ipynb
jupyter notebook segmentation/ultravision_operation/notebook.ipynb
jupyter notebook geometry_and_puzzle_solving/sudoku/notebook.ipynb
jupyter notebook object_tracking/box_factory/main.ipynb
jupyter notebook ocr/packman/main.ipynb
```

---

## 🤝 Team Credits

Developed by **Team Chati Pati** for the **Quera Image Processing Olympiad — Final Round**.
