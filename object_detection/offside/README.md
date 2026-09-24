# Offside Detection — Football Offside Rule Enforcement

> **Competition:** Quera Image Processing Olympiad
> **Problem:** [Quera #355584](https://quera.org/contest/assignments/103147/problems/355584)

---

## Project Overview

**Offside Detection** is a purely classical (non-deep-learning) Computer Vision project that determines whether a football player is in an offside position, given a stylized top-down match diagram. The solution uses color-space segmentation to identify player positions by team color, geometric analysis to find the last defender's line, and pixel-to-metre calibration to report the exact offside distance.

The `offside_detection()` function returns either `"Onside!"` or `"Offside!\n{distance:.2f}"`.

---

## Problem Statement

Given a **top-down football diagram** with:

- An on-screen **legend** showing colored swatches for the two teams (attackers / defenders)
- **Player icons** as colored discs on the pitch
- An indication of which **goal direction** the attackers are facing (top or bottom)

Determine:

1. Whether any attacking player is ahead of the second-to-last defender (the offside line)
2. If yes, how far ahead (in metres) is the most advanced offside player

**Key challenges:**

- White-uniformed teams blend with pitch markings (white lines)
- Multiple players may share the same color
- Images have varying resolutions and pitch orientations

---

## Step-by-Step Solution

### Step 1: Image Normalization — `_normalise_image()`

- Standardizes the input image dimensions and color range.
- Converts the image to a consistent float32 representation for color analysis.

### Step 2: Legend Detection — `_find_legend()`

- Scans the image for the on-screen team legend that shows two colored swatches.
- Returns the bounding box `(left, top, right, bottom)` of the legend area.
- The legend is later masked out from player detection masks to prevent swatch pixels from being detected as players.

### Step 3: Team Color Palette Extraction — `_legend_colours()`

- Reads the dominant colors from each team's swatch in the legend.
- Returns two lists of `(R, G, B)` color tuples, one per team.
- De-duplicates near-identical colors (within 20 intensity units) to avoid redundant masks.

### Step 4: Goal Direction — `_goal_is_at_top()`

- Determines whether the attacking direction is toward the top or bottom goal.
- This determines whether "ahead" means a smaller or larger Y coordinate.

### Step 5: Color Mask Construction — `_build_colour_masks()`

- For each unique team color, creates a **boolean pixel mask** by thresholding the image in color space.
- Pixels whose color distance to the team color is within a tolerance are marked as that color.

### Step 6: Player Detection — `_circle_centres()` and `_coloured_edges()`

**For non-white players (`_coloured_edges`)**:

- Uses `cv2.connectedComponentsWithStats()` to find connected foreground regions in each color mask.
- Filters components by **shape** (height ∈ [12, 34]px, width ∈ [7, 34]px) and **fill ratio** (∈ [0.45, 0.88]) to match player disc geometry.
- Extracts the top or bottom edge of each player disc depending on goal direction.

**For white players (`_circle_centres`)**:

- White pixels overlap with pitch markings, so pure color detection fails.
- Uses a circular template (disc) to detect white blobs with `PLAYER_RADIUS` radius via flood-fill scoring.
- `_classify_white_blob()`: disambiguates which team a white blob belongs to by comparing non-shared palette color counts in the surrounding region — the team with more unique-color pixels nearby wins.

### Step 7: Offside Line Computation

```python
if goal_at_top:
    defender_line = min(team_edges[1])         # Defender closest to attacking goal
    offside_edges = [e for e in team_edges[0] if e < defender_line]
else:
    defender_line = max(team_edges[1])
    offside_edges = [e for e in team_edges[0] if e > defender_line]
```

- The **defender line** is the Y pixel position of the last defender (closest to the goal under attack).
- Any attacker with a Y coordinate beyond the defender line is offside.
- The closest offside attacker is selected; distance is computed as `|defender_line - attacker_edge| × METRES_PER_PIXEL`.

### Step 8: Output

- Returns `"Onside!"` if no attacker is ahead of the defender line.
- Returns `f"Offside!\n{distance:.2f}"` for the distance in metres.

---

## Proposed Enhancements

| Enhancement                                 | Rationale                                                                                                  |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Learned color segmentation (k-NN / GMM)** | More robust to slight color variation and jpeg compression than hard thresholds                            |
| **YOLOv8 player detection**                 | Direct deep detection eliminates all the color-segmentation complexity                                     |
| **Subpixel edge localization**              | Fit a parabola to the Y-gradient profile at each player's edge for sub-pixel accuracy                      |
| **Homographic pitch rectification**         | If the camera angle is not perfectly top-down, a homography step is needed for accurate metre calculation  |
| **Kalman filter tracking (video)**          | Extending to video would allow tracking player positions across frames for real broadcast offside analysis |
| **Multi-referee line detection**            | Auto-detect the second-to-last defender automatically using NMS-style ranking of defender Y positions      |
