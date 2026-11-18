# 🕹️ Packman — Classical Maze Solving, Object Recognition & On-Screen Score OCR

> **Competition:** Quera Image Processing Olympiad  
> **Problem Link:** [Quera #363305 / Assignment #109270](https://quera.org/contest/assignments/109270/problems/363305)

---

## 📖 Project Overview

**Packman** is an integrated competition challenge that bridges classical computer vision, graph-based path planning, and Optical Character Recognition (OCR). Given raw screenshots of vintage Pac-Man maze puzzles, the algorithm must:
1. Detect and rectify the perspective distortion of the game screen.
2. Locate the maze boundary, internal wall graph, and active game entities (Pac-Man, Ghosts, Cherries, Pellets).
3. Read the initial player score displayed on the arcade dashboard using deep OCR engines.
4. Plan an optimal obstacle-free navigation path that maximizes pellet collection within the allocated step budget.
5. Predict the final score and output the move sequence (`U`, `D`, `L`, `R`).

---

## 🛠️ Step-by-Step Technical Pipeline

```
Raw Screen Capture (.png)
          │
          ▼
Blue Mask Thresholding & Contour Quad Detection (`find_maze_quad`)
          │
          ▼
Perspective Warping to Canonical Grid (15 rows x 21 cols, 40px/cell)
          │
          ▼
Wall Grid Confidence & Topological Graph Construction (`build_tree`)
          │
          ▼
Color Segmentation for Game Entities:
  ├─ Yellow Mask ──> Pac-Man & Pellets
  ├─ Red Mask    ──> Ghosts
  └─ Color/Shape ──> Cherry Targets
          │
          ▼
Dashboard OCR Crop ──> PaddleOCR / EasyOCR Initial Score Extraction
          │
          ▼
BFS Shortest-Path / Maximum Pellet Traversal (`bfs_path`)
          │
          ▼
Directional Sequence Generation (`cells_to_str`) ──> submission.csv
```

### 1. Geometric Perspective Rectification
- **Maze Quad Detection (`find_maze_quad`):** Isolates the outer maze walls using blue color thresholding (`blue_mask`), finds convex contours, and extracts the 4 primary corner vertices via polynomial approximation.
- **Ordered Homography (`order_quad`, `warp_perspective`):** Sorts corners into $[TL, TR, BR, BL]$ order and applies a perspective transform to map the maze onto a standardized $840 \times 600$ pixel canvas ($15 \text{ rows} \times 21 \text{ cols}$, $40\text{ px/cell}$).

### 2. Topological Graph Extraction (`build_wall_confidence`)
- Evaluates edge pixel intensity across horizontal and vertical cell boundaries.
- Constructs an undirected adjacency graph $G = (V, E)$ where nodes represent navigable cell centers and edges represent open corridors without blue wall barriers.

### 3. Entity Detection & Identification
- **Pellets & Pac-Man (`yellow_mask`):** Color thresholding in HSV isolates yellow elements; connected component area distinguishes single pellets from the Pac-Man character.
- **Ghosts (`red_mask`):** Identifies hazardous zones to avoid.
- **Bonus Items (`detect_cherry`):** Locates high-value cherries that provide score boosts.

### 4. Score Digit Recognition (OCR)
- Crops the top scoreboard display region.
- Employs dual-engine recognition with fallback: **PaddleOCR** (primary) and **EasyOCR** (secondary).
- Normalizes parsed digits into integer initial scores (`digits_to_score`).

### 5. Path Planning & Metric Optimization
- Executes Breadth-First Search (BFS) / greedy search on the maze corridor graph to find the shortest pellet-dense path connecting starting positions to objective milestones.
- Encodes cell transitions into directional commands: `'U'` (Up), `'D'` (Down), `'L'` (Left), `'R'` (Right).
- Validates path continuity, step limits, and computed pellet harvest efficiency ($>3\text{--}5\text{ pellets per 10 moves}$).

---

## 📂 File Directory

| File / Folder | Purpose |
| :--- | :--- |
| **`main.ipynb`** | End-to-end execution notebook (dataset extraction, CV pipeline, OCR, path solver, validation) |
| **`mini-packman-dataset/`**| Benchmark test set images |
| **`presentation.html`** | Interactive HTML visual presentation of the Pac-Man solution pipeline |
| **`debug/`** | Debug masks and perspective verification visual outputs |
| **`submission.csv`** | Final competition predictions (`case_filename`, `initial_score`, `path`, `yellow_pellets_on_path`, `final_score`) |

---

## 🚀 Quick Start

Run the complete pipeline notebook:
```bash
jupyter notebook main.ipynb
```
