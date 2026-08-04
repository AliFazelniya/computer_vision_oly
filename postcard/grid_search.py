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
    total_ssim = 0.0
    valid_images = 0

    fft_params = {k: params[k] for k in ["mask_height", "mask_width_factor"] if k in params}
    wm_params = {k: params[k] for k in params if k not in ["mask_height", "mask_width_factor"]}

    for c , m , out_name in tests:
        target_path = os.path.join("Watermark", out_name)

        if not os.path.exists(target_path):
            continue

        target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        if target_img is None:
            continue

        try:
            pred_img = apply_watermark_with_params(
                c, m, **fft_params, **wm_params
            )

            score, _ = compute_ssim(target_img, pred_img)

            total_ssim += score
            valid_images += 1

        except Exception:
            pass

    if valid_images > 0:
        return params , total_ssim / valid_images
        
    return params , -1.0

    
def grid_search_ssim_parallel(tests, param_grid, max_workers = 16):
    best_params = None
    best_ssim = -float("inf")
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    total_combos = len(combinations)
    print(f"Total combinations to test: {total_combos}")
    print(f"Using {max_workers} CPU Cores...")

    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers = max_workers) as executor:
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
        