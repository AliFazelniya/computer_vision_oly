# 🧑 Skin Detection — Dual Color-Space Face & Skin Segmentation

> **Competition:** Quera Image Processing Olympiad  
> **Problem Link:** [Quera #292197](https://quera.org/problemset/292197)

---

## 📖 Project Overview

This project implements an illumination-robust skin segmentation algorithm designed to detect human face skin areas in diverse lighting conditions and across different skin tones without requiring heavy neural networks. The method relies on **dual color-space fusion (YCrCb + HSV)** followed by morphological cleaning, connected component filtering, and topological hole filling.

Evaluation is scored by **Intersection over Union (IoU / Jaccard Index)** against benchmark ground-truth segmentation masks.

---

## 🔬 Mathematical & Color-Space Principles

Individual color spaces suffer from characteristic vulnerabilities under real-world lighting:
- **RGB:** Strongly coupled with luminance; skin tone coordinates shift dramatically with changes in light intensity.
- **YCrCb:** Separates luminance ($Y$) from red-difference ($Cr$) and blue-difference ($Cb$) chrominance. Across diverse ethnicities, human skin pigmentation tightly clusters in a narrow $Cr$-$Cb$ ellipse, making it largely invariant to illumination.
- **HSV:** Decouples chromatic tint (Hue) from saturation and lightness (Value). The Hue circular wheel maps red-orange human skin tones to the range $[0^\circ, 50^\circ]$ (in OpenCV: $H \le 25$ or $H \ge 170$).

By computing the **logical intersection (AND)** of both spaces, spurious false-positive detections (such as warm wooden furniture or clay walls) are drastically eliminated.

---

## 🛠️ Step-by-Step Processing Pipeline

```
Input BGR Image
      │
      ▼
Gaussian Blur (5x5, σ=0) ──> Suppress sensor noise & micro-textures
      │
  ┌───┴───────────────────────┐
  ▼                           ▼
BGR ──> YCrCb               BGR ──> HSV
  │                           │
  ├─ Y  > 60                  ├─ (H ≤ 25) ∨ (H ≥ 170)
  ├─ 135 ≤ Cr ≤ 180           ├─ 30 ≤ S ≤ 200
  └─ 85  ≤ Cb ≤ 135           └─ V ≥ 50
  │                           │
  └───────────┬───────────────┘
              ▼
    Logical AND Conjunction
              │
              ▼
Morphological Opening (3x3 Ellipse) ──> Eliminate isolated pixel specks
              │
              ▼
Morphological Closing (7x7 Ellipse, 2 iter) ──> Bridge internal gaps
              │
              ▼
Dilation (3x3 Ellipse, 1 iter) ──> Recover eroded boundary contours
              │
              ▼
Connected Components Analysis (8-way) ──> Retain dominant face component
              │
              ▼
External Contour Hole-Filling (FILLED) ──> Seal eye, eyebrow & mouth shadows
              │
              ▼
Binary Mask Output (IoU Evaluation)
```

---

## 📊 Evaluation & Metrics (`calculate_iou`)

$$\text{IoU} = \frac{|\text{Mask}_{\text{pred}} \cap \text{Mask}_{\text{gt}}|}{|\text{Mask}_{\text{pred}} \cup \text{Mask}_{\text{gt}}|}$$

- Evaluated across test subjects in `./sample/` against ground truth in `./skin_detected/`.
- Predictions are written to `./result/`.

---

## 📂 File Directory

| File / Folder | Description |
| :--- | :--- |
| **`skin.py`** | Complete skin detection, morphological pipeline, and IoU benchmark script |
| **`sample/`** | Test face input photographs (`image_one.jpg` to `image_four.jpg`) |
| **`skin_detected/`** | Official benchmark ground-truth binary masks (`.png`) |
| **`result/`** | Algorithm output masks for verification |

---

## 🚀 Quick Start

Run skin detection and view calculated IoU scores across all samples:
```bash
python3 skin.py
```
