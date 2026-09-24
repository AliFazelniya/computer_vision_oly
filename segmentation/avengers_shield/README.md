# Avengers Shield — Drone Sound Event Segmentation

> **Competition link:** https://quera.org/contest/assignments/103147/problems/355588

Given grayscale **spectrogram images** of mixed drone audio recordings,
detect the precise time intervals during which a specific target drone is active,
and output those intervals in seconds.

---

## Problem Statement

A fleet of drones is deployed for a surveillance mission (*Avengers Shield*).
Audio recordings are captured from the field and converted into **grayscale
spectrogram images** where the horizontal axis represents time and the vertical
axis represents frequency.

Contestants receive:

- **Train set** — spectrogram images + a `train_labels.csv` with semicolon-separated
  `start_time:end_time` intervals (in seconds) per recording.
- **Test set** — spectrogram images without labels.

The goal is to produce a `submission.csv` with one row per test image, each row
containing the semicolon-separated active time intervals for the target drone.

Key challenges:
- Multiple drone types may appear simultaneously in the spectrogram.
- Target drone activity can be sparse, intermittent, or partially overlapping with
  background noise.
- Boundary precision (exact start/end times) is critical for scoring.

---

## Step-by-Step Solution

### Step 1 — Spectrogram Loading & Normalisation

Each image is read with:
```python
spec = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
spec = spec.astype(np.float32) / 255.0   # normalise to [0, 1]
```

The normalised spectrogram `(H, W)` is treated as a 1D temporal sequence of
frequency vectors — i.e., each column is a time-frame feature vector.
This formulation enables **1D temporal convolutional** models to operate
directly on the time axis.

---

### Step 2 — Temporal Convolutional Network (`DroneTCN`)

`DroneTCN` is a **dilated Temporal Convolutional Network** that produces a
frame-level binary activity mask from the raw spectrogram columns.

#### `ConvBlock`

Each building block applies:
```
Conv1d(C_in, C_out, kernel, padding=dilation, dilation=dilation)
  → BatchNorm1d
  → GELU activation
  → Residual connection (1×1 conv if C_in ≠ C_out)
```

GELU activation provides smoother gradients than ReLU and is commonly preferred
in sequence modelling tasks.

#### Dilated Stack

Five stacked `ConvBlock`s with exponentially growing dilation rates:

| Block | Dilation | Receptive field contribution |
|-------|----------|------------------------------|
| 1 | 1 | Local (adjacent frames) |
| 2 | 2 | Short-range context |
| 3 | 4 | Medium-range context |
| 4 | 8 | Long-range context |
| 5 | 16 | Very long-range context |

The exponential dilation schedule gives the network a large effective receptive
field while keeping parameter count linear in the number of blocks — essential
for capturing drone activity patterns that span varying time horizons.

---

### Step 3 — Loss Function

Training uses a **combined BCE + Dice loss**:

```
L = BCE(ŷ, y) + Dice(ŷ, y)
```

- **Binary Cross-Entropy (BCE)** optimises point-wise frame accuracy.
- **Dice loss** directly optimises the overlap between predicted and ground-truth
  activity segments, reducing sensitivity to class imbalance (most frames are
  typically inactive).

---

### Step 4 — Ensemble of Base Models

Three `DroneTCN` instances are trained with:
- **Different convolutional kernel sizes** — varying the per-block receptive field
  and feature granularity.
- **Different random seeds** — ensuring diverse weight initialisations.

At inference, their frame-level probability outputs are combined via
**weighted averaging**, where weights are tuned on the validation fold.
Ensembling reduces variance and smooths spurious high-confidence spikes.

---

### Step 5 — Boundary Refiner (`BoundaryRefiner`)

Raw TCN predictions often have imprecise start/end frame boundaries. A dedicated
`BoundaryRefiner` CNN corrects these using **local spectral context** around each
detected boundary:

- **Input**: a short window of spectrogram columns centred on a candidate boundary frame.
- **Architecture**: a small 1D CNN that regresses a signed boundary offset.
- **Training**: supervised on ground-truth boundary positions extracted from `train_labels.csv`.
- **Cross-validation**: trained with **5-fold cross-validation** and the fold predictions
  are ensembled to produce a single refined boundary estimate.

Three families of refiners (with different loss functions — L1, smooth-L1, robust)
are trained, and their fold models are saved as `refiner_<type>_fold_<N>.pth`.

---

### Step 6 — Post-processing (`mask_bounds`)

Raw frame-level probabilities are converted to a binary activity mask and then
cleaned with morphological operations:

1. **Threshold** — apply a tuned probability threshold to binarise the signal.
2. **Morphological closing** (`cv2.morphologyEx(..., MORPH_CLOSE)`) — merges nearby
   short gaps between active segments; the kernel width controls the *merge gap*
   parameter.
3. **Connected component labelling** (`scipy.ndimage.label`) — enumerates distinct
   contiguous active runs.
4. **Minimum event length filter** — removes spuriously short segments below a
   minimum duration threshold.

---

### Step 7 — Output Formatting

Each connected active segment is converted from frame indices to seconds using
the known frame-rate (time per pixel column). Segments are formatted as
`start_time:end_time` and joined with `;` per image:

```
0.52:1.87;3.14:4.60
```

The final `submission.csv` has one row per test image.

---

## Proposed Enhancements

| Area | Enhancement |
|------|-------------|
| **Input representation** | Use log-mel or Mel-frequency cepstral coefficients (MFCCs) as the spectrogram representation instead of raw grayscale pixels, providing a perceptually motivated frequency axis. |
| **Architecture** | Replace the TCN with a hybrid **CNN-Transformer** (e.g., Conformer) to capture both local patterns and long-range temporal dependencies in a single model. |
| **Multi-scale input** | Feed the model spectrograms at multiple time resolutions (different STFT window sizes) and fuse embeddings — useful when target drone signatures appear at different temporal scales. |
| **Boundary refiner** | Replace the regression-based refiner with a learned **CRF** or **HMM** decoder that models state transitions (active ↔ inactive) to produce globally consistent segmentations. |
| **Data augmentation** | Apply SpecAugment (frequency and time masking), time-stretching, and pitch-shifting to augment the spectrograms, improving robustness to recording condition variability. |
| **Loss** | Add a **boundary-aware loss** term that places extra weight on frames near ground-truth event boundaries, directly encouraging sharp, accurate transitions. |
| **Post-processing** | Replace the fixed threshold with a learned **hysteresis thresholder** (low/high dual-threshold state machine) to reduce false positives from noise bursts near boundaries. |
| **Active learning** | Use model uncertainty (predictive entropy) to flag the most ambiguous test recordings for manual review, improving boundary precision on difficult cases. |
