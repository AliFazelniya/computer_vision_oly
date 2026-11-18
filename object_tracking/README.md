# 🎯 Object Tracking

> **Team:** Chati Pati | **Competition:** Quera Image Processing Olympiad

---

## 📖 Overview

The **Object Tracking** module addresses the challenge of identifying and consistently following multiple moving targets across sequential video frames. Unlike static object detection, multi-object tracking (MOT) requires:
- Robust state estimation in the presence of noise and frame-rate variations.
- Optimal inter-frame data association under occlusions and dense crossovers.
- Accurate track birth, maintenance, and termination lifecycle management.

This directory houses our competition solution for industrial defect tracking ([`box_factory/`](box_factory/)), our comprehensive tracking algorithm curriculum ([`sheet.ipynb`](sheet.ipynb)), and key foundational research papers.

---

## 📂 Subdirectories & Structure

```
object_tracking/
├── box_factory/            # Industrial package defect detection & conveyor tracking (YOLO11s + CenterTracker)
├── sheet.ipynb             # Multi-Object Tracking Laboratory (SORT, Kalman, Hungarian, MOTA)
├── data/                   # Test video streams (people-detection.mp4, conveyor feeds)
├── papers/                 # Research literature links and theoretical references
├── runs/                   # Video tracking visualization and detection outputs
└── yolov8n.pt, yolo26n.pt  # Ultralytics pre-trained detection and segmentation checkpoints
```

---

## 🏆 Featured Projects

### 🏭 [Box Factory](box_factory/)
- **Challenge:** [Quera #310514 / Assignment #109270](https://quera.org/contest/assignments/109270/problems)
- **Problem:** Automated conveyor-belt quality assurance. Detect and track healthy versus defective packages in video footage and count total occurrences without double-counting.
- **Approach:** Fine-tuned `YOLO11s` for package classification paired with custom `CenterTracker` tracking logic with centroid distance gating and persistence buffers.
- **Documentation:** See [`box_factory/README.md`](box_factory/README.md).

---

## 📋 Tracking Curriculum: `sheet.ipynb`

Our 41-code-cell deep-dive explores the evolution of modern Multi-Object Tracking:

| Topic | Theoretical Principle | Implementation Details |
| :--- | :--- | :--- |
| **Centroid Tracking** | Euclidean distance minimization between object centers | Rapid baseline tracker with birth/death lifecycle management |
| **IoU Association** | Geometric bounding box intersection over union | Greedy matching across consecutive bounding boxes |
| **Hungarian Algorithm** | Kuhn-Munkres algorithm for optimal bipartite graph matching | `scipy.optimize.linear_sum_assignment` on cost matrix: $\mathbf{C} = 1 - \text{IoU}$ |
| **Kalman Filter** | Discrete linear quadratic estimation (LQE) | State vector $\mathbf{x} = [c_x, c_y, s, r, \dot{c_x}, \dot{c_y}, \dot{s}]^T$ predicting motion during missed detections |
| **Custom SORT** | Simple Online and Realtime Tracking | Kalman prediction $\rightarrow$ Hungarian matching $\rightarrow$ Kalman measurement update on `people-detection.mp4` |
| **ByteTrack Strategy** | Multi-threshold detection association | Secondary association matching low-confidence detections against unmatched tracks to preserve occluded objects |
| **Evaluation Metrics** | CLEAR MOT benchmark standards | Quantitative calculation of MOTA, MOTP, ID Switches (IDSW), False Positives, and Mostly Tracked (MT) targets |

---

## 📚 Key Research & Literature (`papers/sources.txt`)

- **SORT (Simple Online and Realtime Tracking):** Bewley et al., [arXiv:1602.00763](https://arxiv.org/abs/1602.00763)
- **ByteTrack (Multi-Object Tracking by Associating Every Detection Box):** Zhang et al., [arXiv:2110.06864](https://arxiv.org/abs/2110.06864)
- **Hungarian Algorithm:** Kuhn, H. W., *The Hungarian Method for the Assignment Problem* (1955)
- **Kalman Filtering:** Kalman, R. E., *A New Approach to Linear Filtering and Prediction Problems* (1960)

---

## 🚀 Quick Start

Run the industrial conveyor tracking notebook:
```bash
jupyter notebook box_factory/main.ipynb
```

Run the tracking laboratory notebook:
```bash
jupyter notebook sheet.ipynb
```
