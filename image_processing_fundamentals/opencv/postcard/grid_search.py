"""
grid_search.py — Parallel SSIM-Optimized Hyperparameter Search for Watermarking
================================================================================

This script performs an exhaustive **grid search** over the hyperparameter space
of the distance-transform glow watermarking pipeline (see ``watermark_v2.py``).
It evaluates each parameter combination using **SSIM** (Structural Similarity
Index Measure) as the objective metric and finds the configuration that maximizes
visual similarity between the predicted watermarked image and the competition's
expected output.

Why SSIM instead of MSE / PSNR?
---------------------------------
Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR) treat every
pixel equally, which poorly reflects human visual perception.  SSIM instead
measures three perceptual components jointly:

    SSIM(x, y) = [l(x, y)]^α · [c(x, y)]^β · [s(x, y)]^γ

where:
  • l — luminance similarity: how close the mean brightnesses are
  • c — contrast similarity: how similar the local standard deviations are
  • s — structural similarity: cross-correlation of the local texture patterns

A score of 1.0 is a perfect match; 0.0 means completely dissimilar.

Parallelization strategy
-------------------------
The search space is decomposed into a flat list of all Cartesian-product
combinations using ``itertools.product``.  Each combination is evaluated as
an independent, stateless function call (``evaluate_single_combination``),
making it embarrassingly parallel.  A ``concurrent.futures.ProcessPoolExecutor``
distributes work across multiple CPU cores to reduce wall-clock time.

Parameter groups
-----------------
The grid has two conceptually separate parameter groups that control different
stages of the pipeline:

  1. **FFT parameters** (``mask_height``, ``mask_width_factor``): control how
     aggressively vertical banding is removed from the cover image.
  2. **Watermark parameters** (``radius_px``, ``glow_strength``, etc.): control
     the visual appearance of the distance-transform glow effect.

Usage
-----
    python grid_search.py

Outputs the best parameters and saves final watermarked images to ``pred/``.
"""

import os
import concurrent
import cv2
import numpy as np
import itertools
import concurrent.futures
from skimage.metrics import structural_similarity

def _apply_fft_frequency_mask(fshift: np.ndarray , rows: int , cols: int):
    crow, ccol = rows // 2, cols // 2

    fshift_filtered = fshift.copy()

    mask_height = 1
    mask_width = max(3, cols // 100)

    fshift_filtered[crow - mask_height:crow +
                    mask_height, :ccol - mask_width] = 0
    fshift_filtered[crow - mask_height:crow +
                    mask_height, ccol + mask_width:] = 0

    return fshift_filtered

def _ensure_grayscale(img: np.ndarray):

    if len(img.shape) == 3:
        return cv2.cvColor(img , cv2.COLOR_BGR2GRAY)

    return img.copy()

def remove_vertical_noise(img):

    gray = _ensure_grayscale(img)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape
    fshift_filtered = _apply_fft_frequency_mask(fshift , rows , cols)

    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    cleaned = np.abs(img_back)

    cleaned = cv2.normalize(cleaned, None, 0, 1.0,
                            cv2.NORM_MINMAX).astype(np.float32)

    return cleaned

def _read_gray(path:str ):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Could not read image : {path}")
    return (img.astype(np.float32) / 255.0).clip(0 , 1)

def debug_apply_ref_watermark():
    pass

def _get_distance_maps(msg01: np.ndarray):
    inv = (1.0 - msg01)
    inv_u8 = (np.clip(inv , 0 , 1) * 255).astype(np.uint8)
    _ , hard_u8 = cv2.threshold(inv_u8 , 0 , 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hard = hard_u8.astype(np.uint8)

    di = cv2.distanceTransform(hard, cv2.DIST_L2, 3).astype(np.float32)
    do = cv2.distanceTransform(1 - hard, cv2.DIST_L2, 3).astype(np.float32)

    return hard , di , do

def _get_regions(hard: np.ndarray , di: np.ndarray, do:np.ndarray , radius_px: float, inner_width_px:float):

    outer_region = (hard == 0) & (do > 0) & (do <= radius_px)
    inner_region = (hard == 1) & (di > 0) & (di <= inner_width_px)

    return outer_region , inner_region

def _feather_local(w: np.ndarray, sigma: float, region: np.ndarray):

    if w.max() <= 0:
        return w

    k = max(3 , int(sigma * 3) | 1)

    ww = cv2.GaussianBlur(w, (k ,k) , sigma)
    ww[~region] = 0.0

    return ww

def _compute_intensities(H: int, W: int, do: np.ndarray, di: np.ndarray,
                         outer_region, inner_region, radius_px: float,
                         falloff_power: float, inner_width_px:float, inner_power: float):
    
    glow = np.zeros((H , W), np.float32)
    inn = np.zeros((H , W), np.float32)

    if np.any(outer_region):
        R = float(radius_px)
        p = float(falloff_power)
        glow[outer_region] = np.clip(
            1.0 - (do[outer_region] / R) , 0.0 , 1.0) ** p

    if np.any(inner_region):
        ri = float(radius_px)
        q = float(falloff_power)
        inn[inner_region] = np.clip(
            1.0 - (di[inner_region] / ri) , 0.0 , 1.0) ** q
        
    glow = _feather_local(glow , 0.7 , outer_region)
    inn = _feather_local(inn , 0.6, inner_region)

    return glow , inn

def _blend_effects_to_base(base: np.ndarray, glow: np.ndarray, inn: np.ndarray,
                           outer_region, inner_region,  glow_strength:float , inner_strength: float):

    out = base.copy()

    out[outer_region] = out[outer_region] + glow_strength * glow[outer_region] * (1.0 - out[outer_region])

    out[inner_region] = out[inner_region] - inner_strength * inn[inner_region] * (1.0 - out[inner_region])

    return np.clip(out, 0.0, 1.0).astype(np.float32)

def _apply_ref_watermark(base01: np.ndarray,
                         msg01: np.ndarray,
                         radius_px: float = 4.0,
                         glow_strength: float = 0.95,
                         falloff_power: float = 1.5,
                         inner_width_px: float = 2.8,
                         inner_strength: float = 0.95,
                         inner_power: float = 1.8):

    base = base01.astype(np.float32)
    H , W = base.shape

    hard , di , do = _get_distance_maps(msg01)

    outer_region , inner_region = _get_regions(
        hard , di, do, radius_px, inner_width_px
    )

    glow , inn = _compute_intensities(
        H , W , do , di, outer_region , inner_region,
        radius_px , falloff_power , inner_width_px, inner_power
    )

    out = _blend_effects_to_base(
        base , glow , inn , outer_region , inner_region,
        glow_strength , inner_strength
    )

    return out

def _to_uint8(x: np.ndarray):  
    return (np.clip(x , 0 , 1) * 255.0 + 0.5).astype(np.uint8)

def apply_watermark(cover_name: str, message_name:str):
    cover_path = os.path.join("Data/Cover", cover_name)
    msg_path = os.path.join("Data/Message", message_name)

    cover = _read_gray(cover_path)
    msg = _read_gray(msg_path)

    debanded = remove_vertical_noise(cover)
    out01 = debug_apply_ref_watermark(debanded , msg)

    return _to_uint8(out01)

def _apply_fft_frequency_mask_params(fshift: np.ndarray, rows: int, cols: int,
                                     mask_height: float = 1, mask_width_factor: float = 100):
    crow, ccol = rows // 2, cols // 2

    fshift_filtered = fshift.copy()

    # mask_height from the grid can be fractional (0.5, 2, 3...) but slicing needs an int
    mh = max(1, int(round(mask_height)))
    mask_width = max(3, int(cols // mask_width_factor)) if mask_width_factor else 3

    fshift_filtered[crow - mh:crow + mh, :ccol - mask_width] = 0
    fshift_filtered[crow - mh:crow + mh, ccol + mask_width:] = 0

    return fshift_filtered


def _remove_vertical_noise_with_params(img: np.ndarray, mask_height: float, mask_width_factor: float):
    gray = _ensure_grayscale(img)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape
    fshift_filtered = _apply_fft_frequency_mask_params(
        fshift, rows, cols, mask_height, mask_width_factor
    )

    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    cleaned = np.abs(img_back)

    cleaned = cv2.normalize(cleaned, None, 0, 1.0,
                            cv2.NORM_MINMAX).astype(np.float32)

    return cleaned


def apply_watermark_with_params(cover_name: str, message_name: str,
                                mask_height: float = 1,
                                mask_width_factor: float = 100,
                                radius_px: float = 4.0,
                                glow_strength: float = 0.95,
                                falloff_power: float = 1.5,
                                inner_width_px: float = 2.8,
                                inner_strength: float = 0.95,
                                inner_power: float = 1.8):

    cover_path = os.path.join("Data/Cover", cover_name)
    msg_path = os.path.join("Data/Message", message_name)

    cover = _read_gray(cover_path)
    msg = _read_gray(msg_path)

    debanded = _remove_vertical_noise_with_params(cover, mask_height, mask_width_factor)

    out01 = _apply_ref_watermark(
        debanded, msg,
        radius_px=radius_px,
        glow_strength=glow_strength,
        falloff_power=falloff_power,
        inner_width_px=inner_width_px,
        inner_strength=inner_strength,
        inner_power=inner_power,
    )

    return _to_uint8(out01)

def compute_ssim(target_img: np.ndarray, pred_img: np.ndarray):
    """
    Compute the Structural Similarity Index (SSIM) between two grayscale images.

    SSIM is the primary quality metric for this competition task.  Unlike MSE,
    it accounts for luminance, contrast, and structural information jointly,
    giving a score that better reflects the human perception of image quality.

    The function uses ``skimage.metrics.structural_similarity`` with
    ``data_range=255`` (full 8-bit range) and ``full=True`` to also return
    the per-pixel similarity map (``diff``), which can be used for debugging
    to visualise which regions contribute most to the score.

    Parameters
    ----------
    target_img : numpy.ndarray
        Ground-truth watermarked image (the competition's expected output).
        Must be 2-D (grayscale).
    pred_img : numpy.ndarray
        Predicted watermarked image produced by our pipeline.
        Must be the same shape as ``target_img``.

    Returns
    -------
    score : float
        SSIM score in [0.0, 1.0]; higher is better.
    diff : numpy.ndarray
        Per-pixel difference image in [−1, 1]; used for visualisation.

    Raises
    ------
    ValueError
        If ``target_img`` and ``pred_img`` have different spatial shapes.
    """
    if target_img.shape != pred_img.shape:
        raise ValueError(
            f"Image size mismatch: {target_img.shape} != {pred_img.shape}"
        )

    if target_img.dtype != np.uint8:
        target_img = target_img.astype(np.uint8)

    if pred_img.dtype != np.uint8:
        pred_img = pred_img.astype(np.uint8)

    score, diff = structural_similarity(
        target_img,
        pred_img,
        data_range = 255,
        full = True
    )

    return score, diff

def evaluate_single_combination(params , tests):
    """
    Evaluate a single hyperparameter combination over all test image pairs.

    This function is designed to be called from multiple worker processes
    concurrently (via ``ProcessPoolExecutor``), so it must be **stateless**
    and avoid any global side effects.  It:

    1. Splits ``params`` into FFT parameters and watermark parameters.
    2. For each (cover, message, expected_output) tuple in ``tests``:
       a. Reads the ground-truth watermarked image from ``Watermark/``.
       b. Calls ``apply_watermark_with_params`` with the current parameters.
       c. Computes SSIM between the prediction and the ground truth.
    3. Returns the mean SSIM averaged over all valid test images.

    Parameters
    ----------
    params : dict
        Combined hyperparameter dictionary (keys from ``param_grid``).
    tests : list of tuple (cover_name, message_name, output_name)
        Each tuple identifies one test image triple.

    Returns
    -------
    tuple (params, avg_ssim)
        The input params dict and the mean SSIM.  ``avg_ssim`` is −1.0 if no
        valid images were found (e.g., missing files).
    """
    total_ssim = 0.0
    valid_images = 0

    # Split combined params dict into the two pipeline stages.
    fft_params = {k: params[k] for k in ["mask_height", "mask_width_factor"] if k in params}
    wm_params = {k: params[k] for k in params if k not in ["mask_height", "mask_width_factor"]}

    for c , m , out_name in tests:
        target_path = os.path.join("Watermark", out_name)

        if not os.path.exists(target_path):
            continue

        # Read the expected (ground-truth) watermarked image as grayscale.
        target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        if target_img is None:
            continue

        try:
            # Apply the full watermarking pipeline with the current parameter set.
            pred_img = apply_watermark_with_params(
                c, m, **fft_params, **wm_params
            )

            # Compute SSIM and accumulate.
            score, _ = compute_ssim(target_img, pred_img)

            total_ssim += score
            valid_images += 1

        except Exception:
            # Silently skip any image that errors (e.g., file not found).
            pass

    if valid_images > 0:
        return params , total_ssim / valid_images
        
    return params , -1.0

    
def grid_search_ssim_parallel(tests, param_grid, max_workers = 16):
    """
    Exhaustive parallel grid search over the watermark hyperparameter space.

    Generates all Cartesian-product combinations of the values in ``param_grid``
    using ``itertools.product``, then evaluates each combination in parallel
    using a ``ProcessPoolExecutor`` with up to ``max_workers`` CPU cores.

    The function tracks the globally best SSIM seen so far and prints progress
    updates whenever a new best is found or every 500 combinations.

    Parameters
    ----------
    tests : list of tuple (cover_name, message_name, output_name)
        Test image triples passed to ``evaluate_single_combination``.
    param_grid : dict
        Mapping from parameter name to a list of values to try.
        Example: ``{"radius_px": [3.0, 4.0, 5.0], "glow_strength": [0.8, 0.9]}``.
    max_workers : int, optional
        Number of parallel worker processes.  Default 16.  Set to the number
        of physical CPU cores for optimal throughput.

    Returns
    -------
    best_params : dict or None
        The hyperparameter combination that achieved the highest mean SSIM,
        or ``None`` if no combinations were tested.
    best_ssim : float
        The highest mean SSIM achieved.
    """
    best_params = None
    best_ssim = -float("inf")
    # Build the flat list of all combinations from the Cartesian product.
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    total_combos = len(combinations)
    print(f"Total combinations to test: {total_combos}")
    print(f"Using {max_workers} CPU Cores...")

    completed = 0
    # ProcessPoolExecutor: spawns separate Python interpreter processes so that
    # CPU-intensive work bypasses the GIL and truly parallelizes across cores.
    with concurrent.futures.ProcessPoolExecutor(max_workers = max_workers) as executor:
        # Submit all combinations at once; each future represents one evaluation.
        futures = {executor.submit(evaluate_single_combination , params , tests) : params for params in combinations}

        for future in concurrent.futures.as_completed(futures):
            completed += 1

        params , avg_ssim = future.result()

        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            best_params = params

            print(f"[{completed} / {total_combos}] New Best SSIM: {best_ssim:.4f} | Params : {best_params}")

        if completed % 500 == 0:
            print(f"progress: {completed} / {total_combos} combination")

    return best_params, best_ssim

if __name__ == "__main__":

    tests = [
        ("isfahan.jpg", "Isfahan.jpg", "Isfahan.jpg"),
        ("Kashan.jpg", "Kashan.jpg", "Kashan.jpg"),
        ("Kish.jpg", "Kish.jpg", "Kish.jpg"),
        ("Tehran.jpg", "Tehran.jpg", "Tehran.jpg")

    ]

    param_grid = {
        "mask_height" : [0.5, 1, 2, 3],
        "mask_width_factor" : [1, 2, 3, 5],
        "radius_px" : [4.0],
        "glow_strength" : [0.85, 0.9, 0.95],
        "falloff_power" : [1.5, 2.0, 2.5],
        "inner_width_px" : [2.8, 3.5, 4, 4.5],
        "inner_strength" : [0.7 , 0.8, 0.9],
        "inner_power" : [1.4, 1.5, 1.6]
    }

    print("Starting Parallel Grid Search...")

    best_params, best_ssim = grid_search_ssim_parallel(
        tests, param_grid, max_workers = 16
    )

    if best_params:
        print("\n====================================================")
        print(f"Final Best SSIM: {best_ssim:.4f}")
        print("Best Parameters:")
        for k , v in best_params.items():
            print(f" - {k} : {v}")

        print("\n====================================================")

        fft_best = {k: best_params[k]
                    for k in ["mask_height", "mask_width_factor"]}

        wm_best = {k: best_params[k] for k in best_params if k not in ["mask_height", "mask_width_factor"]}

        os.makedirs("pred", exist_ok = True)

        for c , m , out_name in tests:
            try:
                pred_img = apply_watermark_with_params(c, m, **fft_best, **wm_best)
                cv2.imwrite(os.path.join("pred", out_name), pred_img)

                print(f"Saved final image with best params: {best_params}")
            except Exception as e:
                print(f"[skip] {c}: {e}")
        