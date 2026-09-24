"""
skin.py — Dual Color-Space Skin Detection with IoU Evaluation
==============================================================

This module detects skin (face) regions in color images by fusing two
complementary color-space thresholds:

    • **YCrCb** — separates luminance (Y) from chroma (Cr/Cb), making the
      skin range nearly illumination-invariant.
    • **HSV** — separates hue from saturation and value, giving a compact,
      perceptually meaningful representation of skin color.

Using the logical AND of both masks dramatically reduces false positives:
a pixel must satisfy *both* color-space conditions to be labelled as skin.

Pipeline overview
-----------------
1. Gaussian blur  →  reduce sensor noise before colour analysis
2. YCrCb + HSV conversion  →  project BGR into two skin-friendly spaces
3. Empirical thresholding  →  generate two binary skin masks
4. AND fusion  →  keep only pixels that pass both filters
5. Morphological clean-up  →  remove specks; fill holes
6. Largest connected component  →  isolate the dominant face blob
7. Contour-fill  →  fill interior non-skin pixels (eyes, mouth shadow)
8. IoU evaluation  →  compare final mask against ground-truth

Why dual-space detection?
--------------------------
YCrCb is robust to brightness variation because Y (luma) is separate from Cr
and Cb (chroma). However, under extreme lighting it can still confuse warm-
coloured objects with skin. HSV restricts the hue to a narrow red/orange band
and enforces a minimum saturation, rejecting achromatic and cool-coloured
imposters. Combining the two spaces exploits the strengths of both while
cancelling their individual weaknesses.

References
----------
Kovac, J. et al., "Human Skin Color Clustering for Face Detection," EUROCON 2003.
"""

import cv2
import numpy as np
import os

def _largest_component_mask(mask):
    """
    Keep only the largest connected component of a binary mask.

    After morphological clean-up the skin mask typically contains several
    small, scattered blobs (hands, neck patches, background artefacts) in
    addition to the main face region.  ``cv2.connectedComponentsWithStats``
    labels every contiguous group of foreground pixels and returns per-label
    statistics, including pixel area.  By retaining *only* the label with the
    maximum area we discard all spurious blobs and keep just the dominant face.

    Parameters
    ----------
    mask : numpy.ndarray, dtype=uint8
        Binary mask where 255 = foreground (skin) and 0 = background.

    Returns
    -------
    numpy.ndarray, dtype=uint8
        A mask of the same shape containing only the largest component.

    Notes
    -----
    ``cv2.connectedComponentsWithStats`` uses 8-connectivity, meaning diagonally
    adjacent pixels are considered connected.  Label 0 is always the background;
    foreground labels start at 1.  The ``stats`` matrix has one row per label
    with columns: LEFT, TOP, WIDTH, HEIGHT, AREA.
    """
    # Label every contiguous foreground region; stats[i] holds the bounding-box
    # and pixel count for label i.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels <= 1:
        # No foreground components found
        return mask

    # stats[0] is the background, so we look starting from index 1
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))

    # Build a new mask containing only the pixels belonging to the winner label
    largest_mask = np.zeros_like(mask)
    largest_mask[labels == largest_label] = 255
    return largest_mask


def detect_skin(image_path):
    """
    Detect the skin region of a face in the given image.

    The function applies a **dual color-space** strategy to maximise detection
    precision and recall across varying illumination conditions:

    1. **YCrCb thresholding** — the Cr (red-difference) and Cb (blue-difference)
       channels encode skin colour independently of the luminance channel Y.
       Empirical studies (Kovac 2003, Peer 2003) show that most human skin tones
       cluster tightly in the rectangle Cr ∈ [135, 180] and Cb ∈ [85, 135]
       regardless of race or lighting angle.

    2. **HSV thresholding** — the Hue channel isolates colour tone from
       brightness.  Skin hues map to low values (reds/oranges, H ≤ 25 in
       OpenCV's 0–179 scale) or wrap around the top of the wheel (H ≥ 170).
       The Saturation gate (30–200) rejects near-grey regions (e.g. paper,
       white walls) and over-saturated artefacts.

    3. **AND fusion** — combining both masks retains only pixels that are
       simultaneously in the YCrCb *and* HSV skin ranges, greatly reducing
       false positives.

    Parameters
    ----------
    image_path : str
        Path to the input image.

    Returns
    -------
    numpy.ndarray
        An RGB image with the same width/height as the input image,
        where skin pixels are white (255, 255, 255) and all other
        pixels are black (0, 0, 0).

    Raises
    ------
    FileNotFoundError
        If the image cannot be read from ``image_path``.
    """
    # ---- 1. Read image ----
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    # Reduce high-frequency noise a bit before color analysis.
    # A 5×5 Gaussian kernel smooths sharp pixel transitions caused by
    # sensor noise, JPEG compression artefacts, or specular highlights,
    # which would otherwise produce isolated "skin" pixels in non-skin areas.
    blurred = cv2.GaussianBlur(bgr, (5, 5), 0)

    # ---- 2. Convert to YCrCb and HSV color spaces ----
    # YCrCb decomposes the signal into:
    #   Y  — luminance (brightness), range [0, 255]
    #   Cr — red chrominance difference (R − Y), range [0, 255]
    #   Cb — blue chrominance difference (B − Y), range [0, 255]
    # HSV encodes:
    #   H — hue (dominant wavelength), range [0, 179] in OpenCV
    #   S — saturation (colour purity), range [0, 255]
    #   V — value (brightness), range [0, 255]
    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    Y, Cr, Cb = cv2.split(ycrcb)
    H, S, V = cv2.split(hsv)

    # ---- 3. Threshold in YCrCb space ----
    # Standard empirical skin range in YCrCb.
    # Physiological basis:
    #   • Y > 60 : exclude very dark, under-exposed regions — genuine skin
    #     has at least moderate luminance due to sub-surface scattering.
    #   • Cr ∈ [135, 180] : red-difference chrominance of human skin sits in
    #     this band across all ethnicities; values outside it indicate non-skin
    #     hues (blue, green, purple).
    #   • Cb ∈ [85, 135] : blue-difference is suppressed in skin pixels;
    #     values above 135 indicate cool colours (sky, fabric), and values
    #     below 85 indicate strongly blue-shifted surfaces.
    mask_ycrcb = (
        (Y > 60)
        & (Cr >= 135) & (Cr <= 180)
        & (Cb >= 85) & (Cb <= 135)
    )

    # ---- 4. Threshold in HSV space ----
    # OpenCV's Hue range is [0, 179]. Skin hues roughly fall in
    # low values (near red/orange) or wrap around near 179.
    # • H ≤ 25 or H ≥ 170 : covers warm reds and oranges (0–25°) plus the
    #   wraparound near 360°/179 that also maps to red/pink tones.
    # • S ∈ [30, 200] : excludes achromatic pixels (paper, walls, S < 30)
    #   and highly saturated artificial colours (S > 200).
    # • V ≥ 50 : excludes very dark / shadow regions that cannot be skin.
    mask_hsv = (
        ((H <= 25) | (H >= 170))
        & (S >= 30) & (S <= 200)
        & (V >= 50)
    )

    # ---- 5. Combine both masks ----
    # Logical AND: only pixels satisfying *both* color-space constraints are
    # accepted as skin.  This intersection strategy is the key to the dual-space
    # approach — false positives from each individual mask rarely overlap.
    combined = (mask_ycrcb & mask_hsv).astype(np.uint8) * 255

    # ---- 6. Morphological clean-up ----
    # Structuring elements define the neighbourhood shape for morphological ops.
    # Elliptical elements are preferred for round/organic shapes like faces.
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # MORPH_OPEN = erosion followed by dilation with the same kernel.
    # Geometrically: it slides the small ellipse over the mask and removes any
    # foreground region that cannot fully contain the kernel — effectively
    # deleting isolated specks and thin protrusions smaller than the kernel.
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # MORPH_CLOSE = dilation followed by erosion.
    # Geometrically: it fills concave notches and small holes inside the skin
    # region (e.g. gaps between detected skin patches around the chin or cheek).
    # A larger kernel is used here so that gaps up to ~7 px wide are bridged.
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_big, iterations=2)

    # Slight dilation to recover boundary pixels lost during blurring/thresholding.
    # After the open/close sequence the mask can be slightly smaller than the true
    # skin area; a single-step dilation with the small kernel expands it back.
    cleaned = cv2.dilate(cleaned, kernel_small, iterations=1)

    # ---- 7. Keep only the largest connected component (the face) ----
    # Even after morphological clean-up, multiple disconnected skin blobs may
    # remain (e.g. hands, neck).  We assume the largest blob corresponds to the
    # face, which is typically the most prominent skin region in a portrait.
    final_mask = _largest_component_mask(cleaned)

    # Fill any remaining internal holes (e.g. eyes, mouth shadows).
    # cv2.RETR_EXTERNAL retrieves only the outer contour boundary,
    # and cv2.FILLED paints everything inside that contour solid white,
    # sealing interior dark patches (eyes, brows, open mouth) into the mask.
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(final_mask)
    if contours:
        cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    else:
        filled_mask = final_mask

    # ---- 8. Build the RGB output image ----
    # Create a blank canvas and paint detected skin pixels pure white.
    h, w = filled_mask.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    output[filled_mask == 255] = (255, 255, 255)

    return output

def calculate_iou(pred_path, gt_path):
    """
    Compute the Intersection over Union (IoU) between a predicted skin mask
    and a ground-truth mask.

    IoU (also called the Jaccard Index) is the standard metric for binary
    segmentation quality:

        IoU = |Pred ∩ GT| / |Pred ∪ GT|

    It equals 1.0 for a perfect prediction, 0.0 when the masks share no pixels,
    and is robust to class imbalance because it normalises by the union rather
    than the total image area.

    Parameters
    ----------
    pred_path : str
        File path to the predicted mask image (grayscale or colour).
    gt_path : str
        File path to the ground-truth mask image (grayscale or colour).

    Returns
    -------
    float
        IoU score in the range [0.0, 1.0].  Returns 1.0 when both masks are
        completely empty (vacuously correct prediction).
    """
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)


    # Resize prediction to match ground-truth size if needed.
    # INTER_NEAREST is critical here: bilinear or cubic interpolation would
    # introduce intermediate grey values at mask edges, corrupting binarization.
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Binarize both masks (in case of any anti-aliasing / compression artifacts).
    # A threshold of 127 gives a safe mid-point between 0 and 255.
    pred_mask = pred > 127
    gt_mask = gt > 127

    # Intersection: pixels that are foreground in BOTH masks.
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    # Union: pixels that are foreground in EITHER mask.
    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        # Both masks are completely empty -> define as perfect match
        return 1.0

    # IoU = |intersection| / |union|
    return intersection / union


# ---------------------------------------------------------------------------
# Script entry point — process all sample images and evaluate mean IoU
# ---------------------------------------------------------------------------

SAMPLE_DIR = "./sample/"
SKIN_DIR = "./skin_detected/"
OUT_DIR = "./result/"
os.makedirs(OUT_DIR, exist_ok = True)

sample_images = [
    'image_one.jpg' , 'image_two.jpg' , 'image_three.jpg' , 'image_four.jpg'
]

skin_images = [
    'image_one.png' , 'image_two.png' , 'image_three.png' , 'image_four.png'
]

# Run skin detection on each sample and write the binary result to OUT_DIR
for image in sample_images:
    in_path = SAMPLE_DIR + image
    result = detect_skin(in_path)
    # result is in RGB order; convert to BGR for saving with OpenCV

    out_path = OUT_DIR + image
    cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"Saved output to {out_path}")

# Evaluate IoU for each image against its ground-truth skin mask
ious = []
for image, skin_image in zip(sample_images, skin_images):
    pred_path = OUT_DIR + image
    gt_path = SKIN_DIR + skin_image

    iou = calculate_iou(pred_path, gt_path)
    ious.append(iou)
    print(f"{image}: IoU = {iou:.4f}")

# Report the mean IoU across all test images
mean_iou = sum(ious) / len(ious)
print(f"\nMean IoU over {len(ious)} samples: {mean_iou:.4f}")