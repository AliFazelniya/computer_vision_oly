# 📬 Postcard Watermarking — FFT Denoising & Perceptual Glow Watermarking

> **Competition:** Quera Image Processing Olympiad  
> **Problem Link:** [Quera #307570](https://quera.org/problemset/307570)

---

## 📖 Project Overview

This project tackles a dual-objective classical image restoration and enhancement task on historical Iranian postcards:
1. **Periodic Noise Removal:** Eliminating vertical scanner/sensor banding artefacts from digitized postcard covers in the frequency domain without blurring fine architectural textures.
2. **Perceptual Watermark Embedding:** Overlaying a calligraphic or textual message watermark that looks naturally embossed or backlit, optimizing for maximum **Structural Similarity (SSIM)** against ground-truth benchmarks.

---

## 🛠️ Technical Pipeline & Architecture

### Stage 1: Frequency-Domain Periodic Denoising (`remove_vertical_noise`)
Vertical scanner stripes correspond to a distinct line of high-energy spikes along the horizontal frequency axis in Fourier space:
1. **2D FFT:** Compute the Discrete Fourier Transform with `np.fft.fft2(gray)`.
2. **Spectrum Centering:** Re-center zero-frequency (DC) component using `np.fft.fftshift()`.
3. **Surgical Slab Notch Mask:** Suppress a horizontal stripe of height `2 * mask_height` centered at $(H/2, W/2)$, while preserving a protected central DC guard window of width `mask_width = max(3, cols // mask_width_factor)` to retain global illumination.
4. **Inverse FFT:** Reconstruct spatial domain via `np.fft.ifftshift()` followed by `np.fft.ifft2()`, taking the magnitude and normalizing to $[0, 1]$ float32.

```
Spatial Image (Striped) ──> 2D FFT ──> fftshift ──> Horizontal Notch Filter ──> ifftshift ──> 2D IFFT ──> Denoised Image
```

---

### Stage 2: Watermark Blending Evolution

#### Version 1: Linear Alpha Blending (`watermark_v1.py`)
- Thresholds the watermark message image using an inverted binary threshold (`cv2.THRESH_BINARY_INV`).
- Directly darkens pixel intensities under the mask:
  $$\text{result}[p] = \text{cover}[p] \times (1 - \alpha), \quad \alpha = 0.3$$
- While functional, linear attenuation produces sharp artificial borders and fails to match realistic print or stamp textures.

#### Version 2: Distance-Transform Glow & Inner Shadow (`watermark_v2.py`)
To mimic realistic embossed and backlit stamps, v2 introduces continuous distance fields using Euclidean Distance Transform ($L_2$ norm):
- **Outer Distance Field ($d_o$):** Distance from each background pixel to the closest ink pixel (`cv2.distanceTransform(1 - hard, cv2.DIST_L2, 3)`).
- **Inner Distance Field ($d_i$):** Distance from each ink pixel to the nearest edge (`cv2.distanceTransform(hard, cv2.DIST_L2, 3)`).
- **Power-Law Falloff:**
  $$\text{glow}(p) = \left(1 - \frac{d_o(p)}{R_{\text{outer}}}\right)^{\gamma_{\text{glow}}}, \quad \text{shadow}(p) = \left(1 - \frac{d_i(p)}{R_{\text{inner}}}\right)^{\gamma_{\text{shadow}}}$$
- **Feathering:** Localized Gaussian blur constrained to the active masks to prevent discrete quantization banding.
- **Dynamic Blending:**
  $$\text{out}[p] = \text{base}[p] + s_{\text{glow}} \cdot \text{glow}[p] \cdot (1 - \text{base}[p]) - s_{\text{shadow}} \cdot \text{shadow}[p] \cdot (1 - \text{base}[p])$$

---

## ⚡ Parallel Grid Search Optimization (`grid_search.py`)

A parameter-tuning suite using Python's `concurrent.futures.ProcessPoolExecutor` sweeps through thousands of parameter permutations over CPU cores:
- Evaluates similarity against benchmark targets using **Structural Similarity Index (SSIM)** (`skimage.metrics.structural_similarity`).
- Explores: `mask_height`, `mask_width_factor`, `radius_px`, `glow_strength`, `falloff_power`, `inner_width_px`, `inner_strength`, `inner_power`.
- Automatically outputs the optimal configurations into `pred/` (`isfahan.jpg`, `kashan.jpg`, `kish.jpg`, `tehran.jpg`).

---

## 📂 File Directory

| File | Purpose |
| :--- | :--- |
| **`watermark_v1.py`** | Baseline 2D FFT noise filter + linear $\alpha$-blending watermark |
| **`watermark_v2.py`** | Advanced distance-transform dual-band (glow + shadow) pipeline |
| **`grid_search.py`** | Multi-process parallel hyperparameter search using full SSIM metric |
| **`pred/`** | Generated competition submission outputs |
| **`note1.png`, `note2.png`**| Visual notes and mathematical references for watermark parameters |

---

## 🚀 Quick Start

Run standard v2 watermark embedding:
```bash
python3 watermark_v2.py
```

Run parallel hyperparameter search across multi-core CPU:
```bash
python3 grid_search.py
```
