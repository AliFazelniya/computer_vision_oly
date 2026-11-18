# 🖼️ Image Processing Fundamentals

> **Team:** Chati Pati | **Competition:** Quera Image Processing Olympiad

---

## 📖 Overview

The **Image Processing Fundamentals** module serves as the foundational classical computer vision workbench for **Team Chati Pati**. Before deploying deep learning models, competition problems often demand rigorous deterministic preprocessing, frequency-domain filtering, geometric alignment, and morphological analysis.

This module houses both targeted competition challenge solutions and our comprehensive 7-day OpenCV mastery curriculum.

---

## 📂 Subdirectories & Projects

```
image_processing_fundamentals/
├── postcard/             # FFT noise removal & perceptual watermark embedding
├── skin_finding/         # Dual color-space (YCrCb + HSV) skin segmentation
├── sheet.ipynb           # 7-Day OpenCV Bootcamp: Professional Mastery
└── data/                 # Benchmark images and test assets
```

### 1. 📬 [Postcard Watermarking](postcard/)
- **Challenge:** [Quera #307570](https://quera.org/problemset/307570)
- **Problem:** Remove vertical banding noise from digitized postcards using 2D FFT notch filters, then embed a visible, naturally embossed/glowing watermark message.
- **Techniques:** 2D Fast Fourier Transform (`np.fft.fft2`), spectrum centering (`fftshift`), DC-guarded frequency slab masking, Euclidean Distance Transform ($L_2$ norm) for outer glow & inner shadow, parallel SSIM hyperparameter optimization with `ProcessPoolExecutor`.

### 2. 🧑 [Skin Detection](skin_finding/)
- **Challenge:** [Quera #292197](https://quera.org/problemset/292197)
- **Problem:** Segment face and skin regions under challenging and non-uniform illumination.
- **Techniques:** Dual color-space fusion ($YCrCb \land HSV$), elliptical morphological opening and closing, 8-way connected component analysis to isolate dominant face components, topological hole-filling (`cv2.drawContours(..., FILLED)`), and IoU benchmark scoring.

---

## 📋 Comprehensive Curriculum: `sheet.ipynb`

Our structured 7-day intensive laboratory notebook covers the fundamental techniques of classical computer vision:

| Day | Focus Areas | Key Functions & Concepts |
| :--- | :--- | :--- |
| **Day 1** | Processing Basics, Color Spaces & Channels | `cv2.cvtColor`, channel splitting (`cv2.split`), bitwise masks (`bitwise_and`/`bitwise_or`), color thresholding |
| **Day 2** | Filters, Thresholding, Pyramids & Fourier | Gaussian/Bilateral filters, Otsu & adaptive thresholding, Laplacian pyramids, 2D Discrete Fourier Transform |
| **Day 3** | Gradients & Edge Detection | Sobel ($G_x, G_y$), gradient magnitude/orientation, auto-thresholded Canny edge detector |
| **Day 4** | Morphology & Advanced Geometry | Structuring elements, Top-Hat/Black-Hat transforms, contours, `convexHull`, `minAreaRect`, `fitLine` |
| **Day 5** | Hough Transform | Standard & Probabilistic Hough Lines (`cv2.HoughLinesP`), Hough Circles (`cv2.HoughCircles`) |
| **Day 6** | Geometric Transforms & Matching | Affine & perspective warping (`cv2.warpPerspective`), ORB feature detection, RANSAC homography, multi-scale template matching |
| **Day 7** | Phase 1 Integration Project | Multi-stage pipeline combining bilateral filtering, Canny, HoughLines, and TLS line fitting for robust wall angle estimation |

---

## 🚀 Quick Start

Run the postcard watermarking pipeline:
```bash
python3 postcard/watermark_v2.py
```

Run the skin detection evaluation:
```bash
python3 skin_finding/skin.py
```

Open and run the OpenCV 7-day curriculum:
```bash
jupyter notebook sheet.ipynb
```
