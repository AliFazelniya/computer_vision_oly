# OpenCV Fundamentals — Image Processing Exercises

> **Team:** Chati Pati | **Competition:** Quera Image Processing Olympiad

---

## Overview

The **OpenCV** module serves as the Image Processing Fundamentals workbench for Team Chati Pati. It contains a collection of classical computer vision experiments, preprocessing techniques, and signal processing pipelines that underpin the more advanced deep learning projects throughout the repository.

This module covers four core areas:

- **Postcard Watermarking** — FFT-based noise removal + watermark embedding
- **Skin Detection** — Dual color-space face/skin segmentation
- **OpenCV Exercise Sheet** — Comprehensive edge detection, feature matching, and geometry exercises
- **Video Processing** — Grayscale video conversion pipeline

---

## Sub-Projects

---

### 📬 Postcard Watermarking (`postcard/`)

**Problem**: Given a postcard image with vertical banding noise (caused by scanning artifacts or sensor striping), remove the noise and then embed a visible watermark using a reference message image.

#### `watermark_v1.py` — FFT Noise Removal + Alpha Blending

**Pipeline:**

1. **`remove_vertical_noise(img)`** — Reads the image in grayscale, applies 2D FFT via `np.fft.fft2()`, shifts the zero-frequency component to center with `np.fft.fftshift()`, zeroes out a horizontal band in the frequency domain (which corresponds to vertical periodic noise in the spatial domain), reconstructs via `np.fft.ifftshift()` + `np.fft.ifft2()`, and normalizes back to `[0, 255]`.

2. **`apply_watermark(cover_name, message_name)`** — Reads the cleaned image and the message image, resizes the message to match, binarizes it with `cv2.threshold(..., cv2.THRESH_BINARY_INV)` to create a mask of dark pixels (the watermark region), and applies alpha blending: pixels under the watermark mask are darkened by `alpha=0.3`.

#### `watermark_v2.py` — Distance Transform Glow Effects (Refined Version)

An enhanced version that produces a more visually appealing watermark with soft glow and inner shadow effects using distance transforms:

1. **`_get_distance_maps(msg01)`** — Binarizes the message with Otsu thresholding, then computes:
   - `di` = `cv2.distanceTransform(hard, cv2.DIST_L2, 3)` — distance from each foreground pixel to the nearest background pixel (inner glow)
   - `do` = `cv2.distanceTransform(1 - hard, cv2.DIST_L2, 3)` — distance from each background pixel to the nearest foreground pixel (outer glow)

2. **`_get_regions()`** — Creates boolean masks for the outer glow band (`do ≤ radius_px`) and inner shadow band (`di ≤ inner_width_px`).

3. **`_compute_intensities()`** — Computes power-law falloff: `intensity = (1 - d/R)^falloff_power` for smooth fade.

4. **`_feather_local()`** — Applies `cv2.GaussianBlur()` within each region mask for soft anti-aliased edges.

5. **`_blend_effects_to_base()`** — Additively blends the glow onto the background, subtractively darkens the inner edge.

#### `grid_search.py` — SSIM-Optimized Parameter Search

- Uses `concurrent.futures.ProcessPoolExecutor` for parallel evaluation of all parameter combinations.
- **Metric**: SSIM (Structural Similarity Index) between predicted and expected watermarked images.
- Tests `mask_height`, `mask_width_factor`, `radius_px`, `glow_strength`, `falloff_power`, `inner_width_px`, `inner_strength`, `inner_power` — over thousands of combinations.
- Best parameters are used to generate the final submission images.

---

### 🧑 Skin Detection (`skin_finding/skin.py`)

**Problem**: Detect the skin region of a human face in a given image and produce a binary mask where skin pixels are white.

**Pipeline:**

| Step | Operation             | CV Detail                                                                                                           |
| ---- | --------------------- | ------------------------------------------------------------------------------------------------------------------- |
| 1    | Read & blur           | `cv2.imread()` + `cv2.GaussianBlur((5,5), 0)` — reduces high-frequency noise before color analysis                  |
| 2    | Color conversion      | `cv2.cvtColor(BGR→YCrCb)` + `cv2.cvtColor(BGR→HSV)` — dual color spaces for robustness                              |
| 3    | YCrCb thresholding    | Empirical range: `Cr∈[135,180]`, `Cb∈[85,135]`, `Y>60` — the Cb/Cr axes separate skin from non-skin better than RGB |
| 4    | HSV thresholding      | `H≤25 or H≥170` (red/orange hues), `S∈[30,200]`, `V≥50` — handles skin across diverse lighting                      |
| 5    | Combine masks         | Boolean AND of both masks — reduces false positives                                                                 |
| 6    | Morphological cleanup | MORPH_OPEN (remove noise) → MORPH_CLOSE (fill holes) → DILATE (recover boundary pixels)                             |
| 7    | Largest component     | `cv2.connectedComponentsWithStats()` → keep the largest blob (the face)                                             |
| 8    | Hole filling          | `cv2.findContours()` + `cv2.drawContours(..., FILLED)` — fills eyes/mouth shadows                                   |
| 9    | Output                | Binary RGB image: skin=white `(255,255,255)`, background=black `(0,0,0)`                                            |

**Evaluation**: IoU (Intersection over Union) against ground-truth mask images.

---

### 📋 OpenCV Exercise Sheet (`sheet1.ipynb`)

A comprehensive educational notebook covering advanced OpenCV operations:

| Exercise                | Key Functions                                                   | Concept                                                                     |
| ----------------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Sobel Gradients         | `cv2.Sobel()`, `cv2.cartToPolar()`                              | Edge detection via gradient magnitude and angle                             |
| Auto-Canny              | `cv2.Canny()` with median-based thresholds                      | Adaptive edge detection                                                     |
| Top-Hat Transform       | `cv2.morphologyEx(MORPH_TOPHAT)`                                | Background normalization / text extraction under uneven illumination        |
| Bounding Boxes          | `cv2.boundingRect()`, `cv2.minAreaRect()`, `cv2.convexHull()`   | Axis-aligned vs. rotated vs. convex object bounds                           |
| Robust Line Fitting     | `cv2.fitLine(DIST_L1)`                                          | Outlier-resistant geometric fitting                                         |
| Hough Circles           | `cv2.HoughCircles(HOUGH_GRADIENT)`                              | Circular object detection                                                   |
| ORB + RANSAC Homography | `cv2.ORB_create()`, `cv2.findHomography(RANSAC)`                | Feature-based image alignment                                               |
| Wall Angle Estimation   | `cv2.bilateralFilter()` + `cv2.HoughLinesP()` + `cv2.fitLine()` | Expert pipeline combining noise removal, edge detection, and robust fitting |

---

### 🎬 Video Processing (`video_gray.ipynb`)

A minimal but complete video processing pipeline:

| Step             | API                                        | Detail                                     |
| ---------------- | ------------------------------------------ | ------------------------------------------ |
| Open video       | `cv2.VideoCapture()`                       | Decodes MP4 using the system's video codec |
| Read metadata    | `cap.get(CAP_PROP_FPS)` etc.               | Frame rate, resolution, total frames       |
| Frame display    | `clear_output()` + `display()`             | Inline video playback in Jupyter           |
| Color conversion | `cv2.cvtColor(BGR→GRAY)`                   | Luminance: `Y = 0.299R + 0.587G + 0.114B`  |
| Write output     | `cv2.VideoWriter()` + `fourcc('mp4v')`     | Grayscale video saved as MP4               |
| Cleanup          | `cap.release()` + `video_writer.release()` | Flush codec buffers and close file handles |

---

## Proposed Enhancements

### Watermarking

- Replace SSIM with **LPIPS (Learned Perceptual Image Patch Similarity)** for more perceptually accurate optimization
- Explore **invisible frequency-domain watermarking** (DWT-based) for steganographic applications

### Skin Detection

- Use **face detection (dlib/MediaPipe)** to first localize the face, then apply color segmentation within that ROI
- Replace empirical thresholds with a **learned Gaussian Mixture Model (GMM)** on skin color distributions

### Video Processing

- Add **optical flow** (`cv2.calcOpticalFlowPyrLK`) for motion estimation between frames
- Integrate **background subtraction** (`cv2.createBackgroundSubtractorMOG2`) for foreground object detection
