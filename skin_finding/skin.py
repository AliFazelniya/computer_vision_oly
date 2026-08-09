import cv2
import numpy as np
import os

def _largest_component_mask(mask):
    """
    Keep only the largest connected component of a binary mask.
    This helps to isolate the face region and remove small
    false-positive skin blobs scattered around the image.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels <= 1:
        # No foreground components found
        return mask

    # stats[0] is the background, so we look starting from index 1
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))

    largest_mask = np.zeros_like(mask)
    largest_mask[labels == largest_label] = 255
    return largest_mask


def detect_skin(image_path):
    """
    Detect the skin region of a face in the given image.

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
    """
    # ---- 1. Read image ----
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    # Reduce high-frequency noise a bit before color analysis
    blurred = cv2.GaussianBlur(bgr, (5, 5), 0)

    # ---- 2. Convert to YCrCb and HSV color spaces ----
    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    Y, Cr, Cb = cv2.split(ycrcb)
    H, S, V = cv2.split(hsv)

    # ---- 3. Threshold in YCrCb space ----
    # Standard empirical skin range in YCrCb
    mask_ycrcb = (
        (Y > 60)
        & (Cr >= 135) & (Cr <= 180)
        & (Cb >= 85) & (Cb <= 135)
    )

    # ---- 4. Threshold in HSV space ----
    # OpenCV's Hue range is [0, 179]. Skin hues roughly fall in
    # low values (near red/orange) or wrap around near 179.
    mask_hsv = (
        ((H <= 25) | (H >= 170))
        & (S >= 30) & (S <= 200)
        & (V >= 50)
    )

    # ---- 5. Combine both masks ----
    combined = (mask_ycrcb & mask_hsv).astype(np.uint8) * 255

    # ---- 6. Morphological clean-up ----
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Remove small noise
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small, iterations=1)
    # Close small holes inside the skin region
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_big, iterations=2)
    # Slight dilation to recover boundary pixels lost during blurring/thresholding
    cleaned = cv2.dilate(cleaned, kernel_small, iterations=1)

    # ---- 7. Keep only the largest connected component (the face) ----
    final_mask = _largest_component_mask(cleaned)

    # Fill any remaining internal holes (e.g. eyes, mouth shadows)
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(final_mask)
    if contours:
        cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    else:
        filled_mask = final_mask

    # ---- 8. Build the RGB output image ----
    h, w = filled_mask.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    output[filled_mask == 255] = (255, 255, 255)

    return output

def calculate_iou(pred_path, gt_path):
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)


    # Resize prediction to match ground-truth size if needed
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Binarize both masks (in case of any anti-aliasing / compression artifacts)
    pred_mask = pred > 127
    gt_mask = gt > 127

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        # Both masks are completely empty -> define as perfect match
        return 1.0

    return intersection / union


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

for image in sample_images:
    in_path = SAMPLE_DIR + image
    result = detect_skin(in_path)
    # result is in RGB order; convert to BGR for saving with OpenCV

    out_path = OUT_DIR + image
    cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"Saved output to {out_path}")

ious = []
for image, skin_image in zip(sample_images, skin_images):
    pred_path = OUT_DIR + image
    gt_path = SKIN_DIR + skin_image

    iou = calculate_iou(pred_path, gt_path)
    ious.append(iou)
    print(f"{image}: IoU = {iou:.4f}")

mean_iou = sum(ious) / len(ious)
print(f"\nMean IoU over {len(ious)} samples: {mean_iou:.4f}")