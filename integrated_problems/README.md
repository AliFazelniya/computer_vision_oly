# 🧩 Integrated Problems & Competition Preparation

> **Team:** Chati Pati | **Competition:** Quera Image Processing Olympiad — Final Round Phase 5

---

## 📖 Overview

The **Integrated Problems** directory represents the capstone phase of our preparation for the Quera Image Processing Olympiad. Real competition challenges rarely fall into isolated categories; instead, they demand complex multi-stage pipelines that unify:
- **Geometric Analysis:** Line estimation, coordinate transforms, perspective rectification, and Total Least Squares (TLS) fitting.
- **Object Detection & Tracking:** Fast object localization, Kalman filter state estimation, and Hungarian-algorithm data association.
- **Complex OCR & Language Engineering:** Non-Latin script (Persian) Unicode normalization, font rendering, degradation simulation, and OCR engine integration.
- **Synthetic Data Generation & Stress Testing:** Automated parametric test-case synthesis to rigorously benchmark algorithms against controlled failure modes (illumination, blur, occlusions, perspective distortion).
- **Time-Constrained Contest Simulation:** Strict 60-minute mock contest runner harness.

---

## 📂 Directory Structure

```
integrated_problems/
├── notebook.ipynb                # Comprehensive Phase 5 Master Integration Notebook (42 code cells)
├── competition-prep/             # Production-grade, modular Python CV toolbox with self-tests
│   ├── cv_basics/                # Image/video I/O, drawing, timing, logging utilities
│   ├── detection/                # Bounding box utilities, IoU, NMS, classical text-region detectors
│   ├── tracking/                 # Constant-velocity BoxKalman tracker, line-crossing event counter
│   ├── geometry/                 # Line normal forms, signed distance, angle folding, TLS line fitting
│   ├── ocr/                      # Persian text normalization, contrast/deskew preprocessing, CER metric
│   ├── evaluation/               # Line accuracy, MOT metrics (MOTA/MOTP), detection PR curves
│   ├── synthetic_data/           # Parametric generators for wall edges, crossing videos, Persian plates
│   ├── mock_contests/            # 60-minute contest simulation harness and starter templates
│   └── run_all_tests.py          # Unified test runner executing self-tests across all modules
└── integrated_problems/          # Generated evaluation datasets, benchmark reports, and video outputs
    ├── mock_contest/             # Hidden evaluation challenge sets
    ├── reports/                  # Systematic factor-sweep stress test reports (.csv)
    ├── text/                     # Synthesized Persian text dataset samples
    ├── videos/                   # Annotated synthetic line-crossing tracking videos (.mp4)
    └── walls/                    # Dev and unseen wall boundary benchmark images
```

---

## 🔬 Core Components & Technical Deep-Dive

### 1. Master Integration Notebook (`notebook.ipynb`)
A 42-code-cell laboratory notebook establishing the complete end-to-end curriculum:
1. **Wall-Edge Benchmark & Stress Sweeps:** Generates synthetic indoor floor/wall intersection scenes with controlled degradation factors (noise, blur, illumination gradients, floor clutter). Implements custom metric `score_wall` measuring angular error (degrees modulo $180^\circ$) and orthogonal distance error (pixels), with automated heatmap factor-sweep reporting.
2. **Robust Geometric Line Fitting:** Normal form $ax + by + c = 0$, undirected angle folding into $(-90^\circ, 90^\circ]$, point-line orthogonal projection, line clipping to rectangular viewports, and Total Least Squares (TLS) regression via Singular Value Decomposition (SVD).
3. **Multi-Object Tracking & Line Crossing:** Synthesizes multi-object video trajectories with occlusion and missed detections. Implements `BoxKalman` state filter ($[c_x, c_y, w, h, v_x, v_y, v_w, v_h]$) paired with Hungarian assignment (`scipy.optimize.linear_sum_assignment`), and directional line-crossing counters for traffic/conveyor monitoring.
4. **Persian OCR & Text Processing:** Handles right-to-left (RTL) text, zero-width non-joiners (ZWNJ `\u200c`), digit unification (Persian $\leftrightarrow$ Arabic $\leftrightarrow$ Latin), text rendering via PIL/Raqm, automated contrast stretching, deskewing, and robust Tesseract integration with Levenshtein-based Character Error Rate (CER).

---

### 2. Competition Prep Reusable Toolbox (`competition-prep/`)
Every module in `competition-prep/` is completely standalone, dependency-isolated, and contains its own self-test:
- **`cv_basics/timing.py`:** Precision context-manager `Timer` and `ContestClock` to guard against timeout disqualifications.
- **`geometry/lines.py`:** Self-contained line algebra and clipping algorithms.
- **`tracking/kalman_tracker.py`:** Pure NumPy Kalman filter implementation with covariance propagation.
- **`ocr/normalize.py`:** Unicode NFKC normalization and canonical evaluation policies for Persian text.
- **`run_all_tests.py`:** Executes each module in a clean subprocess to ensure modularity.

---

### 3. The 60-Minute Contest Protocol (`mock_contests/`)
Our team's standardized operational procedure for competition problems:

| Time | Phase | Operational Objective |
| :---: | :--- | :--- |
| **00 – 05 min** | Task Triage | Deconstruct problem statement, verify submission schema, and write metric function |
| **05 – 15 min** | Baseline Harness | Implement a minimal, crash-safe baseline (`safe_method`) to guarantee non-zero score |
| **15 – 40 min** | Iterative Optimization | Address the single largest failure mode identified on validation errors, one change at a time |
| **40 – 52 min** | Generalization Check | Validate on held-out synthetic test set; add deterministic fallbacks for edge cases |
| **52 – 58 min** | Freeze & Full Run | Freeze code logic, execute full test run, log execution timings |
| **58 – 60 min** | Format Verification | Inspect output CSV/JSON schema, column headers, shape, and null values |

---

## 🚀 Quick Start

Run the complete self-test suite across all competition-prep modules:
```bash
python3 competition-prep/run_all_tests.py
```

Test a specific module directly (e.g., line geometry):
```bash
python3 competition-prep/geometry/lines.py
```

Simulate a 60-minute mock contest session:
```bash
python3 competition-prep/mock_contests/template_solution.py
```
