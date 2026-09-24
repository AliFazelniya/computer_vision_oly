# Football Analysis — YOLO Player Detection & Tracking

> **Competition:** [Quera Image Processing Olympiad](https://quera.org/problemset/307571)

---

## Project Overview

**Football Analysis** is an object detection project focused on detecting football players in broadcast match images. The solution leverages **YOLOv8x** (the extra-large variant), the state-of-the-art anchor-free real-time object detector, to identify player bounding boxes with high confidence. Post-processing includes K-Means color clustering for team assignment and statistical analysis of player positions.

---

## Problem Statement

Given a dataset of football broadcast images, detect all visible players and output their bounding box coordinates. Additional goals include:

- Filtering out non-player detections (referees, advertising boards)
- Handling varying image scales and perspectives
- Optionally clustering players by team color

**Key challenges:**

- Players partially occlude each other
- Wide variety of pitch angles (aerial, side-line, close-up)
- Background noise: crowd, goal posts, pitch markings

---

## Step-by-Step Solution

### Step 1: Environment Setup & Configuration

```python
MODEL_NAME = "yolov8x.pt"       # Extra-large YOLO variant for maximum accuracy
PLAYER_CONFIDENCE = 0.2          # Low threshold — catch borderline players
MIN_AREA_RATIO = 0.0002          # Filter out tiny false positives
MAX_AREA_RATIO = ...             # Filter out unrealistically large detections
```

- A low confidence threshold (0.2) is intentional: players in the background are small and produce weak confidence scores. Post-filtering by bounding box area removes false positives.

### Step 2: Image Discovery

- **`list_image_files(root)`**: Recursively searches the `data/` directory for `.jpg`, `.jpeg`, `.png` files using `Path.rglob()`.
- Handles nested competition dataset structures automatically.

### Step 3: Model Loading & Inference

```python
model = YOLO("yolov8x.pt")
results = model(image_path, conf=PLAYER_CONFIDENCE)
```

- **YOLOv8x**: The 68M parameter variant of YOLOv8 provides the best detection accuracy at the cost of speed — appropriate for competition use where inference time is not critical.
- Outputs per-image list of detections: `(class_id, x_center, y_center, width, height, confidence)`.

### Step 4: Detection Filtering

- Filters YOLO detections to the **"person"** class only (class ID 0 in COCO).
- Area ratio filter: `(w * h) / (img_w * img_h)` must be within `[MIN_AREA_RATIO, MAX_AREA_RATIO]` to reject: very small blobs (noise), and very large blobs (partial camera/crowd views).

### Step 5: Team Color Clustering (K-Means)

- Crops each player bounding box.
- Extracts the dominant jersey color using `sklearn.cluster.KMeans(n_clusters=2)` in HSV space.
- Groups detections into two clusters (Team A / Team B) based on color centroids.

### Step 6: Results & Submission

- Records all accepted detections with coordinates and confidence scores.
- Outputs a structured CSV for submission.

---

## Proposed Enhancements

| Enhancement                                     | Rationale                                                                                 |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **YOLOv8x fine-tuning on football datasets**    | Pre-training on general COCO leaves room for improvement on football-specific appearances |
| **DeepSORT / ByteTrack tracking**               | Adds temporal consistency — assigns consistent player IDs across frames                   |
| **Homographic pitch mapping**                   | Project player positions onto a canonical top-down pitch view for tactical analysis       |
| **Faster team assignment via palette matching** | Replace K-Means per-crop with a shared palette estimated from the legend area             |
| **NMS tuning (IoU threshold)**                  | Reduce missed detections of overlapping players by adjusting NMS overlap threshold        |
| **RT-DETR or DETA**                             | Transformer-based detectors have better precision on occluded/small objects               |
| **SAM (Segment Anything Model)**                | Segment exact player silhouettes for pixel-level team jersey analysis                     |
