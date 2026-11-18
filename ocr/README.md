# 🟣 Optical Character Recognition (OCR)

> **Team:** Chati Pati | **Competition:** Quera Image Processing Olympiad

---

## 📖 Overview

The **Optical Character Recognition (OCR)** category addresses the translation of visual text imagery into machine-readable digital characters. In this competition, OCR challenges span multiple complex sub-domains:
- **Degraded Handwritten Math Recognition:** Segmenting multi-stroke expressions and classifying mixed digits and arithmetic operators under extreme noise.
- **Embedded Game Screen OCR:** Reading low-resolution arcade scoreboard digits in stylized fonts under geometric perspective distortion.
- **Persian / RTL Script Engineering:** Tackling complex non-Latin script challenges including right-to-left layout, cursive character shaping, Zero-Width Non-Joiners (ZWNJ), font rendering engines (Raqm), and Persian Tesseract engine optimization.

---

## 📂 Subdirectories & Projects

```
ocr/
├── ugly_handwriting/       # Deep Learning: Multi-stroke handwriting segmentation & 16-class CNN
├── packman/                # Integrated: Maze perspective rectification, entity detection & scoreboard OCR
├── phase4_persian_ocr/     # Persian OCR Research: Calibration sets, multiline documents & engine shootout
└── sheet.ipynb             # 60-Cell Master Persian OCR Laboratory Notebook
```

---

## 🏆 Featured Projects

### 1. ✍️ [Ugly Handwriting](ugly_handwriting/)
- **Challenge:** Quera Image Processing Olympiad
- **Problem:** Recognize degraded, smudged handwritten mathematical expressions consisting of digits `0-9` and operators (`+`, `-`, `×`, `÷`, `(`, `)`).
- **Architecture:** Two-stage pipeline:
  1. Style classifier (`best_type_classifier.keras`) predicting handwriting format (print vs. cursive).
  2. Multi-box vertical merging (`merge_vertical_boxes`) and padding/normalization to $28 \times 28$ pixels.
  3. 16-class deep convolutional network with class-balanced cross-entropy loss.
- **Documentation:** See [`ugly_handwriting/README.md`](ugly_handwriting/README.md).

### 2. 🕹️ [Packman](packman/)
- **Challenge:** [Quera #363305 / Assignment #109270](https://quera.org/contest/assignments/109270/problems/363305)
- **Problem:** Solve visual Pac-Man maze puzzles: rectify perspective, detect walls and food items, read scoreboard digits with deep OCR (PaddleOCR / EasyOCR), and compute optimal pellet-collection paths using BFS graph traversal.
- **Documentation:** See [`packman/README.md`](packman/README.md).

### 3. 🇮🇷 [Phase 4 Persian OCR Research](phase4_persian_ocr/)
- **Problem:** End-to-end Persian document analysis and engine benchmarking.
- **Key Artifacts:**
  - `calibration/`: Test images evaluating edge cases (clean, cut-off, heavy noise, hard contrast).
  - `multiline/`: Multi-line Persian document images with ground-truth transcriptions (`labels.csv`).
  - `synthetic50/`: 50 parametrically degraded Persian plate and sign images with full label coverage.
  - `reports/`: Comparative benchmark reports (`day25_pipeline_results.csv`, `day26_ocr_results.csv`, `day31_all_results.csv`, `ext_engine_shootout.csv`).

---

## 📋 Curriculum: `sheet.ipynb`

Our comprehensive 60-code-cell Persian OCR laboratory provides an exhaustive reference for non-Latin script processing:
1. **Persian Unicode & Typography:** Handling Unicode code points, bidirectional embedding (Bidi), glyph shaping, and canonical Zero-Width Non-Joiner (ZWNJ `\u200c`) normalization.
2. **Font Discovery & Raqm Rendering:** Automated discovery of system fonts supporting Persian glyphs (`پچژگکی۰۱۲۳۴۵۶۷۸۹`), dynamic shaping via FreeType/Raqm, and synthetic text layout.
3. **Synthetic Degradation Modeling:** Simulating document degradation: uneven illumination backgrounds, salt-and-pepper noise, motion/defocus blur, and random affine skew.
4. **Adaptive Image Preprocessing:** Robust contrast stretching (`dynamic_range` percentiles), adaptive thresholding, polarity detection, and automated deskewing.
5. **Crash-Safe Engine Integration:** Sandboxed Tesseract execution (`fas` language model) with automatic cropping, upscaling, and padding.
6. **Error Evaluation Metrics:** True Levenshtein edit distance and Character Error Rate (CER) calculation on canonical text forms.

---

## 🚀 Quick Start

Run handwriting recognition:
```bash
jupyter notebook ugly_handwriting/main.ipynb
```

Run the Pac-Man maze solving and OCR pipeline:
```bash
jupyter notebook packman/main.ipynb
```

Explore the Persian OCR master notebook:
```bash
jupyter notebook sheet.ipynb
```
