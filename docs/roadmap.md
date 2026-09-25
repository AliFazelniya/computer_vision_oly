# 🏆 Image Processing Competition — Final Preparation

> 43-day intensive preparation plan for the in-person final.
> Daily time: 4–5 hours
> Format: Image-based tasks + provided datasets
> Team: Same team as preliminary round
> Main topics: Object Detection & Tracking, Wall-edge Detection / Geometry, Persian OCR
> Final duration: 4.5 hours / 270 minutes
> YOLO model: Provided during the final

---

## 🎯 Final Goal

Become able to take an unfamiliar image-processing problem, inspect its dataset, design a baseline, implement the pipeline quickly, debug systematically, and produce the required output within 270 minutes.

### Core skill tree

- [ ] OpenCV & classical image processing
- [ ] Geometry & line detection
- [ ] YOLO / Object Detection
- [ ] Object Tracking
- [ ] Persian OCR
- [ ] Detection + Tracking + Geometry
- [ ] Detection + OCR
- [ ] Competition-time coding
- [ ] Team workflow
- [ ] Full 270-minute mock contests

---

# 🗺️ 43-Day Roadmap

| Phase | Days | Focus |
|---|---:|---|
| Phase 1 | 1–7 | OpenCV, Geometry, Wall Edge Detection |
| Phase 2 | 8–15 | YOLO & Object Detection |
| Phase 3 | 16–23 | Object Tracking |
| Phase 4 | 24–31 | Persian OCR |
| Phase 5 | 32–38 | Integrated Problems |
| Phase 6 | 39–43 | Mock Finals & Final Readiness |

---

# 📚 Permanent Resource Hub

## OpenCV

- Official docs: https://docs.opencv.org/4.x/
- PyImageSearch: https://pyimagesearch.com/
- Kaggle datasets: https://www.kaggle.com/datasets

## Detection

- Ultralytics Docs: https://docs.ultralytics.com/
- Roboflow Universe: https://universe.roboflow.com/
- COCO: https://cocodataset.org/

## Tracking

- SORT: https://arxiv.org/abs/1602.00763
- ByteTrack: https://arxiv.org/abs/2110.06864
- MOTChallenge: https://motchallenge.net/
- FilterPy: https://filterpy.readthedocs.io/en/latest/

## OCR

- Tesseract: https://tesseract-ocr.github.io/
- PaddleOCR: https://www.paddleocr.ai/
- Kaggle: https://www.kaggle.com/datasets
- Persian OCR / ALPR examples:
  - https://github.com/anajar2198/persian-lpr-yolov11
  - https://alitourani.github.io/pages/project-vehicle-alpr.html

## Dataset search strategy

### For OpenCV / Geometry
Search Kaggle or Roboflow for:
- `lane detection`
- `wall detection`
- `edge detection`
- `line detection`
- `road lane`

### For Object Detection
Search:
- `person detection`
- `object detection`
- `custom object detection`
- `wall detection`

### For Tracking
Use:
- MOTChallenge
- MOT17
- Kaggle video datasets
- Short self-recorded videos

### For Persian OCR
Search:
- `Persian OCR`
- `Farsi OCR`
- `Persian text recognition`
- `Persian handwritten`
- `Iranian license plate`

> Practice data should be small enough to iterate quickly. The objective is problem-solving, not downloading huge datasets.

---

# 🧠 Daily Study Protocol

Use this structure for most 4–5 hour sessions:

| Block | Time | Activity |
|---|---:|---|
| Theory | 30–45m | Learn only the day's concepts |
| Implementation | 90m | Implement from scratch |
| Challenge | 90m | Solve a new problem / dataset |
| Debug & evaluation | 30–45m | Analyze failures and record metrics |

### Daily rule

> Every day must produce a measurable artifact: code, benchmark, visualization, evaluation, or solved task.

---

# 🟦 Phase 1 — OpenCV & Geometry

## Day 1 — Image Processing Basics

### Learn
- BGR / RGB
- HSV
- Grayscale
- Resize
- Crop / ROI
- Histogram
- Pixel manipulation

### Practice
- Select 5 images from Kaggle / COCO.
- Implement `resize_keep_ratio()`.
- Extract multiple ROIs.
- Compare RGB/BGR/HSV representations.
- Generate and compare histograms.

### Deliverable
A reusable OpenCV basics notebook.

### Pass criteria
- [ ] 5+ reusable functions written independently
- [ ] Correct channel / shape handling
- [ ] No unexplained copied code

---

## Day 2 — Filtering & Thresholding

### Learn
- Gaussian Blur
- Median Blur
- Bilateral Filter
- Otsu Threshold
- Adaptive Threshold

### Practice
Generate:
- Gaussian noise
- Salt-and-pepper noise

Test several filters on the same images.

### Resource
https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html

### Evaluation
- [ ] Explain which filter fits which noise
- [ ] Compare edge preservation
- [ ] Produce a visual comparison report

---

## Day 3 — Edge Detection

### Learn
- Sobel
- Scharr
- Laplacian
- Canny

### Practice dataset
Use 10–20 images containing walls, roads, objects, or architectural edges from Kaggle.

Test:
- Multiple Canny thresholds
- Sobel vs Canny
- Noisy vs clean images

### Resource
https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html

### Evaluation
- [ ] Understand Canny pipeline
- [ ] Explain threshold effects
- [ ] Identify false edges

---

## Day 4 — Morphology & Contours

### Learn
- Erosion
- Dilation
- Opening
- Closing
- Morphological Gradient
- Contours
- Connected Components

### Practice
Generate 20 synthetic binary images containing:
- Small noise
- Holes
- Broken objects
- Multiple shapes

Extract:
- bbox
- area
- centroid
- aspect ratio

### Resource
https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html

### Evaluation
Create:

`extract_objects(mask) -> [{bbox, area, centroid, aspect_ratio}]`

---

## Day 5 — Hough Lines & Geometry

### Learn
- Hough Transform
- HoughLinesP
- Line representation
- Line angle
- Line intersection
- Point-to-line distance

### Practice
Use lane-detection images from Kaggle or synthetic images.

Implement:
- line filtering
- angle calculation
- intersection
- point-line distance

### Resource
https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html

### Evaluation
- [ ] 20+ geometry unit tests
- [ ] Correct edge cases
- [ ] Numerical error measured

---

## Day 6 — Perspective & Wall Edge Detection

### Learn
- Homography
- Perspective Transform
- ROI
- Line fitting

### Practice dataset
Create your own synthetic wall dataset:
- Different edge angles
- Different backgrounds
- Shadows
- Noise
- Broken edges
- Distractor lines

Compare:
- Canny + Hough
- Contours
- Line fitting

### Evaluation
Measure:
- angle error
- distance error
- success rate

Target:
- ≥90% success on your controlled test set

---

## Day 7 — Phase 1 Project

### Project
**Wall / Reference Line Detection**

Pipeline:

`Image → Preprocess → Edge → Hough/Contour → Filter → Geometry → Output`

### Deliverables
- Annotated images
- Detected line
- Angle
- Distance
- Error analysis

### Pass criteria
Solve a new sample from scratch in ≤90 minutes.

---

# 🟩 Phase 2 — YOLO & Object Detection

## Day 8 — Detection Fundamentals

### Learn
- Classification vs Detection
- Bounding Boxes
- IoU
- Confidence
- Precision / Recall
- NMS

### Practice
Use 20 COCO images.

Implement IoU from scratch with NumPy.

Test:
- identical boxes
- no overlap
- partial overlap
- edge touching
- containment

### Evaluation
At least 30 IoU unit tests.

---

## Day 9 — YOLO Output Mastery

### Learn
- YOLO architecture at a practical level
- Backbone
- Neck
- Detection Head
- Confidence
- Classes
- Bounding boxes
- NMS

### Practice
Run YOLO on 10 images.

Extract into your own format:

```python
{
    "bbox": [x1, y1, x2, y2],
    "confidence": 0.91,
    "class_id": 0
}
```

Test confidence thresholds:
- 0.2
- 0.5
- 0.8

### Resource
https://docs.ultralytics.com/modes/predict/

### Evaluation
Build `detect_objects()` independently.

---

## Day 10 — Detection Pipeline

### Practice
Build:
- image input
- inference
- class filtering
- confidence filtering
- visualization
- standardized output

### Robustness tests
- empty image
- no detections
- very small object
- multiple objects
- unexpected image size

### Pass criteria
Pipeline handles all cases without crashing.

---

## Day 11 — Detection Dataset

### Dataset
Choose a small Roboflow Universe dataset with bounding boxes.

https://universe.roboflow.com/

### Practice
- Read annotations
- Visualize boxes
- Inspect bad annotations
- Identify small objects
- Calculate basic precision / recall

### Evaluation
Explain at least 5 failure cases.

---

## Day 12 — Classical Detection

### Project
Colored-object detection.

Pipeline:

`Image → HSV → Mask → Morphology → Connected Components → BBoxes`

### Data
Generate 50 synthetic images with:
- circles
- rectangles
- different colors
- noise
- different sizes

### Evaluation
Compare classical detection against YOLO.

---

## Day 13 — Detection Benchmark

### Practice
For the same 50-image synthetic dataset:
- measure Precision
- measure Recall
- measure inference time
- analyze false positives

### Deliverable
One benchmark table.

---

## Day 14 — Detection Project

### Project
YOLO object detection pipeline.

### Tasks
- dataset setup
- inference
- visualization
- confidence tuning
- false-positive analysis

### Evaluation
Connect a pretrained detector to a new problem in ≤30 minutes.

---

## Day 15 — Detection Stress Test

Test your pipeline with:
- lighting variation
- blur
- scale changes
- clutter
- partial objects

### Evaluation
Write a one-page failure analysis.

---

# 🟨 Phase 3 — Object Tracking

## Day 16 — Centroid Tracking

### Learn
- Centroids
- Distance matching
- Track IDs
- New / lost objects

### Data
Use:
- a short self-recorded video
- a simple Kaggle video

### Build
Centroid tracker from scratch.

### Pass criteria
Correct IDs on simple scenes.

---

## Day 17 — IoU Tracking

### Learn
- IoU matrix
- Greedy matching
- Match threshold

### Practice
Connect YOLO detections to your IoU tracker.

### Evaluation
Test different IoU thresholds and document ID switches.

---

## Day 18 — Hungarian Matching

### Learn
- Assignment problem
- Cost matrix
- Hungarian algorithm

### Practice
Use:

`scipy.optimize.linear_sum_assignment`

Compare:
- greedy matching
- Hungarian matching

### Evaluation
Construct at least 10 synthetic matching cases.

---

## Day 19 — Kalman Filter

### Learn
- State
- Measurement
- Prediction
- Update
- Process noise
- Measurement noise

### Practice
Simulate a moving point with noisy measurements.

### Resource
https://filterpy.readthedocs.io/en/latest/

### Evaluation
- [ ] Predict through missing measurements
- [ ] Explain state and measurement
- [ ] Visualize predicted vs actual trajectory

---

## Day 20 — Track Lifecycle

Implement:

```python
class Track:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
```

Support:
- create
- update
- predict
- delete
- track buffer

### Evaluation
Simulate objects entering and leaving the scene.

---

## Day 21 — YOLO + Tracker

### Pipeline

`Frame → YOLO → Matching → Track Update → Visualization`

Track:
- ID
- bbox
- trajectory
- confidence history
- last seen

### Evaluation
Record:
- ID switches
- missed tracks
- false tracks
- track fragmentation

---

## Day 22 — Tracking Project

### Project
Multi-object tracking on real video.

### Dataset
MOTChallenge / MOT17:
https://motchallenge.net/

### Goal
Run your own tracker on a short sequence.

---

## Day 23 — Tracking Comparison

Compare your tracker with:
- SORT
- ByteTrack

Resources:
- https://arxiv.org/abs/1602.00763
- https://arxiv.org/abs/2110.06864

### Evaluation
For at least 5 failure cases, identify whether the problem is:
- detection
- association
- prediction
- track lifecycle

---

# 🟥 Phase 4 — Persian OCR

## Day 24 — OCR Fundamentals

### Learn
- Text Detection
- Text Recognition
- OCR pipeline
- RTL text
- connected Persian script

### Practice
Create 50 synthetic Persian text images using multiple fonts.

Vary:
- font size
- font
- spacing
- background
- rotation

---

## Day 25 — Persian OCR Preprocessing

### Learn
- grayscale
- Otsu
- adaptive threshold
- denoising
- deskew
- morphology
- contrast enhancement

### Practice
Run 3+ preprocessing pipelines on the 50-image set.

### Evaluation
Measure which preprocessing works best for which image conditions.

---

## Day 26 — Tesseract

### Learn
- Persian language model
- PSM
- confidence
- bounding boxes

### Resource
https://tesseract-ocr.github.io/

### Practice
Run OCR on your 50-image dataset.

### Evaluation
For 10 failures, identify the cause:
- preprocessing
- text detection
- recognition
- normalization

---

## Day 27 — Persian Normalization

### Implement
- Arabic/Persian character normalization
- `ي → ی`
- `ك → ک`
- whitespace normalization
- Persian/English digits
- zero-width non-joiner handling

### Evaluation
Build a normalization test suite with at least 30 cases.

---

## Day 28 — Text Region Detection

### Learn
- contours
- connected components
- horizontal projection
- vertical projection
- line segmentation
- word regions

### Practice
Create multi-line Persian images.

### Goal
Extract text regions before OCR.

---

## Day 29 — Character Recognition

### Practice
Train a small CNN on a suitable digit / character dataset.

If a suitable Persian dataset is unavailable:
- use digits / Latin characters for architecture practice
- continue Persian OCR pipeline with Tesseract

### Evaluation
Produce:
- validation accuracy
- confusion matrix
- failure examples

---

## Day 30 — OCR Evaluation

### Learn
- Character Accuracy
- Word Accuracy
- CER
- Levenshtein Distance
- substitutions
- deletions
- insertions

### Implement
CER from scratch.

### Formula

`CER = (S + D + I) / N`

### Evaluation
Test on at least 30 reference/prediction pairs.

---

## Day 31 — Persian OCR Project

### Pipeline

`Image → Preprocess → Text Detection → Crop → OCR → Normalize → Output`

### Dataset
Use your synthetic Persian dataset + real Persian images from Kaggle.

Search:
- Persian OCR
- Farsi OCR
- Persian text recognition
- Iranian license plate

### Evaluation
- ≥50 test images
- Ground Truth available
- Analyze ≥10 failures
- Report CER

---

# 🟪 Phase 5 — Integrated Problems

## Day 32 — Wall Edge Stress Test

### Dataset
Synthetic + real images.

Include:
- shadows
- noise
- broken edges
- distractor lines
- different angles

### Metrics
- angle error
- distance error
- success rate
- runtime

---

## Day 33 — Wall Edge Final Pipeline

Create a robust version:

`Input → Preprocess → Candidate Edges → Line Detection → Filtering → Geometry → Output`

### Challenge
Solve unseen images in ≤60 minutes.

---

## Day 34 — Detection + Tracking + Line

### Project
Detect objects and determine whether they cross a reference line.

### Pipeline

`YOLO → Tracking → Line Geometry → Crossing Detection`

### Practice
Use:
- self-recorded video
- MOT/Kaggle video

### Evaluation
Prevent double-counting.

---

## Day 35 — Tracking Stress Test

Add:
- occlusion
- missed detections
- multiple similar objects
- fast movement

### Metrics
- ID switches
- missed tracks
- false tracks
- crossing accuracy

---

## Day 36 — Detection + OCR

### Project
Detect text regions, crop them, OCR them, normalize output.

Possible scenarios:
- product labels
- signs
- license plates

### Reference projects
https://github.com/anajar2198/persian-lpr-yolov11
https://alitourani.github.io/pages/project-vehicle-alpr.html

---

## Day 37 — OCR Stress Test

Vary:
- blur
- rotation
- low contrast
- small text
- clutter
- multiple text regions

### Evaluation
Measure CER before and after preprocessing.

---

## Day 38 — Competition Code Engineering

Create:

```text
competition-prep/
├── cv_basics/
├── detection/
├── tracking/
├── geometry/
├── ocr/
├── evaluation/
├── synthetic_data/
└── mock_contests/
```

### Build reusable utilities
- image loading
- visualization
- bbox utilities
- IoU
- geometry
- timing
- logging
- OCR normalization
- evaluation

### Evaluation
Every utility should be runnable independently.

---

# 🏁 Phase 6 — Mock Finals

## Day 39 — Mock Contest #1

### Time: 270 minutes

| Task | Time |
|---|---:|
| Problem analysis | 20m |
| Dataset inspection | 20m |
| Pipeline design | 20m |
| Baseline implementation | 90m |
| Debugging | 60m |
| Final improvement/output | 60m |

### Rule
Use an unfamiliar problem.

### Post-mortem
Record:
- time lost
- bugs
- wrong assumptions
- missing utilities
- team communication issues

---

## Day 40 — Detection + Tracking Mock

### Scenario
Unfamiliar video / image sequence.

Must include:
- Detection
- Tracking
- IDs
- geometry or temporal information

### Evaluation
- correctness
- runtime
- robustness
- output format

---

## Day 41 — OCR + Geometry Mock

### Scenario
Persian text + geometric requirement.

Must include:
- preprocessing
- ROI extraction
- OCR
- normalization
- geometry

### Evaluation
- OCR quality
- geometry error
- runtime
- robustness

---

## Day 42 — Full Team Mock Final

### Conditions

- 270 minutes
- Same team
- Unseen dataset
- Unseen problem
- Same working environment
- No tutorial videos
- Only allowed documentation/resources
- Full output submission

### Required Post-mortem

| Category | Score / Notes |
|---|---|
| Problem understanding | |
| Dataset inspection | |
| Baseline speed | |
| Coding speed | |
| Debugging | |
| Detection | |
| Tracking | |
| OCR | |
| Geometry | |
| Team communication | |
| Time management | |
| Final output correctness | |

---

## Day 43 — Final Readiness

### Technical checklist

- [ ] IoU from scratch
- [ ] NMS concept
- [ ] YOLO inference
- [ ] Confidence filtering
- [ ] Bounding box manipulation
- [ ] Canny
- [ ] Morphology
- [ ] Contours
- [ ] Hough
- [ ] Point-line distance
- [ ] Perspective transform
- [ ] Centroid tracking
- [ ] IoU tracking
- [ ] Hungarian matching
- [ ] Kalman Filter
- [ ] Track lifecycle
- [ ] OCR preprocessing
- [ ] Tesseract / available OCR pipeline
- [ ] Persian normalization
- [ ] CER
- [ ] Detection + OCR
- [ ] Detection + Tracking
- [ ] Tracking + Geometry

### Environment checklist

- [ ] Python environment tested
- [ ] Required packages installed
- [ ] YOLO model execution tested
- [ ] OpenCV tested
- [ ] OCR stack tested
- [ ] Dataset loading tested
- [ ] Output writing tested
- [ ] Runtime measurement available
- [ ] Team workflow tested

### Final rule

Do not learn a major new topic today.

Prioritize:
- confidence
- speed
- stability
- sleep
- environment readiness

---

# ⏱️ 270-Minute Final Strategy

## 0–30 min — Understand

- Read every task
- Inspect dataset
- Understand input/output
- Identify scoring
- Estimate difficulty

## 30–60 min — Baseline

- Run provided YOLO
- Inspect detections
- Test a few samples
- Establish the simplest working pipeline

## 60–170 min — Build

- Implement core logic
- Integrate detection
- Add geometry / tracking / OCR
- Keep code modular

## 170–230 min — Debug

- Test edge cases
- Visualize intermediate outputs
- Inspect failures
- Fix highest-impact issues first

## 230–270 min — Submission

- Validate output
- Check filenames / format
- Run final pipeline from scratch
- Confirm no missing files

---

# 🤝 Team Workflow

## Suggested responsibilities

| Area | Primary | Backup |
|---|---|---|
| Problem analysis | Member 1 | Member 2 |
| YOLO / Detection | Member 2 | Member 1 |
| Tracking | Member 2 | Member 3 |
| Geometry / Wall Edge | Member 3 | Member 1 |
| OCR | Member 3 | Member 2 |
| Integration | Everyone | — |
| Debugging | Everyone | — |

### Important

No critical component should belong to only one person.

Each teammate must be able to:
- run the main pipeline
- understand the code
- modify thresholds
- debug common failures
- produce the final output

---

# 📊 Weekly Milestones

## End of Week 1
- [ ] Strong OpenCV baseline
- [ ] Geometry utilities
- [ ] Wall-edge prototype

## End of Week 2
- [ ] YOLO output mastery
- [ ] Detection pipeline
- [ ] IoU / NMS understanding

## End of Week 3
- [ ] Custom tracker
- [ ] YOLO + tracker
- [ ] Tracking failure analysis

## End of Week 4
- [ ] Persian OCR pipeline
- [ ] Normalization
- [ ] CER evaluation

## End of Week 5
- [ ] Integrated CV pipelines
- [ ] Detection + Tracking
- [ ] Detection + OCR
- [ ] Wall-edge robust solution

## End of Week 6
- [ ] At least 2 full mock contests
- [ ] Team workflow tested
- [ ] 270-minute time management tested

---

# 🧪 Competition Readiness Scorecard

Score yourself 0–2 for each item:

- 0 = cannot do independently
- 1 = can do with documentation
- 2 = can do quickly from memory

| Skill | Score |
|---|---:|
| OpenCV preprocessing | /2 |
| Thresholding | /2 |
| Morphology | /2 |
| Contours | /2 |
| Canny | /2 |
| Hough | /2 |
| Geometry | /2 |
| Perspective transform | /2 |
| IoU | /2 |
| NMS | /2 |
| YOLO inference | /2 |
| Detection post-processing | /2 |
| Centroid tracking | /2 |
| IoU tracking | /2 |
| Hungarian matching | /2 |
| Kalman filter | /2 |
| Track lifecycle | /2 |
| OCR preprocessing | /2 |
| Persian normalization | /2 |
| OCR evaluation | /2 |
| Detection + OCR | /2 |
| Detection + Tracking | /2 |
| Debugging | /2 |
| Dataset inspection | /2 |
| Team workflow | /2 |
| 270-minute execution | /2 |

### Target

**Before the final: ≥45 / 52**

---

# 🔥 Core Principle

> **Do not optimize for how much you study. Optimize for how fast you can turn an unfamiliar image-processing problem into a working, testable pipeline.**

The final skill you are training is:

`Problem → Dataset → Hypothesis → Baseline → Code → Debug → Evaluate → Improve → Submit`

That loop should become automatic by Day 43.
