# Sudoku Solver — Project Overview

> **Competition link:** https://quera.org/contest/assignments/103147/problems/355590

Given a **photograph** of a printed Sudoku puzzle, automatically detect the board,
extract and classify each digit cell with a lightweight CNN, then produce a complete,
valid solution using constraint-propagation backtracking.

---

## Problem Statement

The publisher *Memar Zehen* released a large collection of printed Sudoku books.
Contestants are given:

- **Train set** — JPEG photographs of Sudoku pages + ground-truth CSV grids (81 digit values, 0 = empty).
- **Test set** — JPEG photographs without labels.

The task is to output a `submission.csv` where each row contains the `puzzle_id`
followed by the 81 solved cell values in reading order (row-major, 1-indexed columns).

Key challenges:
- Images are taken at arbitrary angles, distances, and lighting conditions.
- Printed digits share visual similarity (e.g., 1 vs 7, 3 vs 8).
- A single mis-classified digit can make an otherwise valid board unsolvable.

---

## Step-by-Step Solution

### Step 1 — Image Preprocessing & Board Localisation

| Operation | API | Purpose |
|-----------|-----|---------|
| Grayscale read | `cv2.imread(..., IMREAD_GRAYSCALE)` | Discard colour channels |
| Gaussian blur | `cv2.GaussianBlur(img, (7,7), 0)` | Attenuate high-frequency noise |
| Adaptive threshold | `cv2.adaptiveThreshold(..., ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2)` | Binarise with local contrast normalisation |
| Contour detection | `cv2.findContours(thresh, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` | Extract outer contours |
| Largest contour | `max(contours, key=cv2.contourArea)` | Isolate the board boundary |
| Convex hull | `cv2.convexHull(c)` | Smooth irregular outline |
| Polygon approx. | `cv2.approxPolyDP(hull, 0.03 * arcLength, True)` | Reduce to 4 corner points |
| Perspective warp | `getPerspectiveTransform` + `warpPerspective` | Rectify board → 450×450 px |

`order_points()` sorts the four detected corners into *[top-left, top-right, bottom-right, bottom-left]* order using the sum/difference heuristic, ensuring a consistent destination mapping.

---

### Step 2 — Cell Extraction (`extract_cells`)

The 450×450 warped board is divided into a **9×9 grid** of 50×50 px tiles.
For each tile:

1. Crop with a **6-pixel margin** on all sides (removes grid lines → 38×38 region).
2. **Invert** pixel values: `(255 − pixel) / 255` — digits become bright on a dark background, matching MNIST-style convention.
3. **Resize** to **28×28** via `cv2.resize`.

Output: NumPy array of shape `(81, 1, 28, 28)`, float32, values in `[0, 1]`.

---

### Step 3 — Digit Classifier (`SudokuCNN`)

A compact CNN trained **from scratch** to classify each cell into one of
**10 classes** (0 = empty, 1–9 = printed digit).

```
Input  (1 × 28 × 28)
  │
  ├─ Conv2d(1→32, 3×3, pad=1) + BN + ReLU
  ├─ Conv2d(32→64, 3×3, pad=1) + BN + ReLU
  ├─ MaxPool2d(2×2)                          → 64 × 14 × 14
  │
  ├─ Conv2d(64→128, 3×3, pad=1) + BN + ReLU
  ├─ MaxPool2d(2×2)                          → 128 × 7 × 7
  │
  ├─ Flatten → 6272-d
  ├─ FC(6272→128) + ReLU
  ├─ Dropout(p=0.3)
  └─ FC(128→10)   [logits]
```

- **Batch Normalisation** after each convolution stabilises gradient flow.
- **Dropout(0.3)** before the output head reduces overfitting on the cell-image dataset.
- **Ensemble**: Two models are trained with different random seeds (42, 123); their softmax outputs are averaged at inference to reduce prediction variance.

---

### Step 4 — Data Augmentation

Printed digits have low intra-class variation, so augmentation is intentionally mild
but targets the main real-world distortions:

- **Affine transform** (`cv2.getRotationMatrix2D` + `cv2.warpAffine`):
  - Rotation: ±6°
  - Scale: ±8%
  - Translation: ±2 px
- **Brightness jitter**: pixel values multiplied by a uniform factor in [0.85, 1.15], then clipped to [0, 1].

**Critical design choice in `AugmentedDataset`:** augmentation is applied **only to non-empty cells** (`label != 0`). Empty cells are deliberately kept clean so the model learns to distinguish a blank cell from a faint digit.

---

### Step 5 — Constraint-Propagation Solver

#### `check_board_valid(grid)`
O(81) scan that checks the three Sudoku constraints (all rows, columns, 3×3 boxes
contain no repeated non-zero values). Used as a fast guard before launching the solver.

#### `get_candidates(grid, r, c)`
Returns the list of valid digits for cell `(r, c)` by collecting values already
present in the same row, column, and 3×3 box and returning the complement.

#### `solve_sudoku(grid)` — MRV Backtracking
Implements the **Minimum-Remaining-Values (MRV)** heuristic:
at each recursive step, the empty cell with the *fewest legal candidates* is chosen
next. This dramatically prunes the search tree compared to a naïve left-to-right scan.
The function mutates the grid in-place and backtracks on failure.

---

### Step 6 — Constraint-Guided Decoding (`decode_with_solver`)

Bridges CNN uncertainty with Sudoku logic using a cascade of increasingly expensive
correction strategies:

| Stage | Strategy |
|-------|----------|
| 0 | **Greedy decode** (`argmax`) → attempt direct solve |
| 1 | **1-substitution search** — for the 25 most uncertain cells, try each runner-up digit and re-solve |
| 2 | **2-substitution search** — pairwise correction over the 12 most uncertain cells |
| 3 | **Fallback** — iteratively zero-out the least-confident conflicting clue and re-solve |

Cell uncertainty is measured as the *softmax probability gap* between the top-1 and
top-2 predicted classes. Cells with a small gap are considered ambiguous and are
prioritised for substitution.

---

### Step 7 — Training & Inference

**Training (`train_model`)**

- Optimiser: **AdamW** (lr=1e-3, weight_decay=1e-4) — decoupled weight decay
  prevents L2 regularisation from being absorbed by the adaptive moment scaling.
- Scheduler: **CosineAnnealingLR** anneals the learning rate to ~0 over 25 epochs,
  avoiding abrupt LR drops.
- Batch size: 64 with `shuffle=True`.

**Inference pipeline (per test image)**

1. `get_warped(path)` → 450×450 grayscale board.
2. `extract_cells(warped)` → `(81, 1, 28, 28)` tensor.
3. Forward pass through both ensemble models; average softmax outputs.
4. `decode_with_solver(probs)` → `(initial_grid, solved_grid)`.
5. If solver fails, fill remaining zeros with `1` as a safe fallback.
6. Append row to `submission.csv`.

---

## Proposed Enhancements

| Area | Enhancement |
|------|-------------|
| **Board detection** | Replace contour heuristics with a learned corner detector (e.g., fine-tuned CornerNet or YOLO-based board localiser) for robustness to extreme perspective and partial occlusion. |
| **Cell classifier** | Pre-train on a merged MNIST + printed-digit dataset; use MobileNetV3-Small for a better accuracy/speed trade-off. |
| **Augmentation** | Add elastic distortions, JPEG compression artefacts, and shadow simulation to better match scan/photo quality variation. |
| **Ensemble** | Increase to 5 models with diverse architectures (CNN + ViT-Tiny) and aggregate with learned ensemble weights. |
| **Decoding** | Extend `decode_with_solver` with k=3 substitutions; use LP relaxation to find the globally most-probable valid board instead of greedy cascade. |
| **End-to-end training** | Fine-tune the full pipeline with a differentiable Sudoku solver layer (e.g., SATNet) so the CNN loss directly reflects board-level correctness. |
| **Uncertainty calibration** | Apply temperature scaling on the CNN logits so the softmax gaps used in `decode_with_solver` are better calibrated probability estimates. |
