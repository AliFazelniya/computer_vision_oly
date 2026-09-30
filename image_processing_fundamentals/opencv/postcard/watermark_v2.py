"""
watermark_v2.py — Refined FFT Noise Removal + Distance-Transform Glow Watermarking
====================================================================================

This module is the **refined successor** to ``watermark_v1.py``.  It retains
the same FFT-based vertical banding removal pipeline but replaces the simple
alpha-blend watermark with a visually richer **distance-transform glow effect**
that mimics the appearance of an embossed or back-lit stamp.

The glow effect has two concentric zones around every watermark stroke:

    1. **Outer glow** — pixels *outside* the watermark ink region that are
       within ``radius_px`` pixels of the nearest ink stroke are brightened.
       The brightening decays with distance (power-law falloff), creating a
       soft halo reminiscent of neon or back-lighting.

    2. **Inner shadow** — pixels *inside* the ink region that are within
       ``inner_width_px`` pixels of the nearest ink–background boundary are
       darkened.  This gives the stroke a three-dimensional edge-shadow look.

Both effects are post-processed with a localised Gaussian blur (feathering)
to avoid hard quantisation bands at the falloff boundaries.

Module structure
----------------
_apply_fft_frequency_mask   — zero out banding frequencies in shifted spectrum
_ensure_grayscale           — colour→grey helper
remove_vertical_noise       — full FFT denoise pipeline
_read_gray                  — safe float32 grayscale reader
_get_distance_maps          — compute inner/outer distance fields from msg mask
_get_regions                — boolean masks for outer glow and inner shadow bands
_feather_local              — Gaussian feathering confined to a region
_compute_intensities        — power-law falloff → glow and shadow weight maps
_blend_effects_to_base      — additively apply glow/shadow onto the base image
_apply_ref_watermark        — orchestrate the full glow watermark pipeline
_to_uint8                   — convert float32 [0,1] to uint8 [0,255]
apply_watermark             — public entry point: read files + run full pipeline

See Also
--------
watermark_v1.py — simpler version using only alpha blending.
"""

import os
import cv2
import numpy as np

def _apply_fft_frequency_mask(fshift: np.ndarray , rows: int , cols: int):
    """
    Zero out horizontal-banding frequency components in a shifted FFT spectrum.

    After ``np.fft.fftshift`` the spectrum's DC (zero-frequency) component sits
    at position (rows//2, cols//2).  Vertical banding in the spatial image
    appears as peaks along the centre row of the spectrum (they have zero
    vertical spatial frequency).  This helper zeroes a narrow horizontal slab
    of height 2·``mask_height`` centred on that row, excluding a small
    ``mask_width``-wide guard around the DC column so the image mean is preserved.

    Parameters
    ----------
    fshift : numpy.ndarray (complex128)
        FFT spectrum after ``np.fft.fftshift``.
    rows : int
        Number of rows in the original image (= spectrum height).
    cols : int
        Number of columns in the original image (= spectrum width).

    Returns
    -------
    numpy.ndarray (complex128)
        Filtered spectrum with banding coefficients zeroed out.
    """
    crow, ccol = rows // 2, cols // 2  # Centre (DC) position in shifted spectrum

    fshift_filtered = fshift.copy()

    # Height of the suppression slab above/below the centre row.
    # A value of 1 gives a total slab of 2 rows, which is sufficient for
    # typical scanner banding that is concentrated at a single frequency row.
    mask_height = 1
    # Guard width around DC: at least 3 columns, or 1 % of image width.
    # This protects the DC component (image average brightness) from being
    # zeroed, which would cause a dramatic mean-brightness shift in the output.
    mask_width = max(3, cols // 100)

    # Zero the left half of the banding slab (negative horizontal frequencies).
    fshift_filtered[crow - mask_height:crow +
                    mask_height, :ccol - mask_width] = 0
    # Zero the right half (positive horizontal frequencies; conjugate symmetric
    # partner required to keep the reconstructed image real-valued).
    fshift_filtered[crow - mask_height:crow +
                    mask_height, ccol + mask_width:] = 0

    return fshift_filtered

def _ensure_grayscale(img: np.ndarray):
    """
    Return a grayscale copy of ``img``.

    Parameters
    ----------
    img : numpy.ndarray
        Input image — either BGR (3-channel) or already single-channel.

    Returns
    -------
    numpy.ndarray, dtype same as input
        Single-channel (H, W) array.
    """
    if len(img.shape) == 3:
        return cv2.cvColor(img , cv2.COLOR_BGR2GRAY)

    return img.copy()

def remove_vertical_noise(img):
    """
    Remove vertical banding artefacts from a grayscale image via FFT filtering.

    The function performs a full forward FFT → frequency-domain mask → inverse
    FFT cycle.  See ``_apply_fft_frequency_mask`` for the mask rationale and
    ``watermark_v1.remove_vertical_noise`` for a more detailed explanation of
    the frequency-domain intuition.

    Unlike the v1 implementation, this function:
    • Accepts only grayscale input (colour conversion is the caller's
      responsibility via ``_ensure_grayscale``).
    • Returns a **float32 image normalised to [0, 1]** rather than uint8,
      so subsequent processing can work entirely in floating-point without
      precision loss from repeated quantisation rounds.

    Parameters
    ----------
    img : numpy.ndarray, dtype=uint8 or float32
        Grayscale input image.

    Returns
    -------
    numpy.ndarray, dtype=float32, values in [0, 1]
        De-banded grayscale image ready for blending operations.
    """
    gray = _ensure_grayscale(img)

    # 2-D forward FFT: transform the spatial image into the frequency domain.
    # Each complex coefficient F(u,v) represents the amplitude and phase of a
    # spatial sinusoid oscillating at u cycles per row and v cycles per column.
    f = np.fft.fft2(gray)

    # Shift the zero-frequency component to the array centre for easier masking.
    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape
    # Apply the horizontal-banding suppression mask.
    fshift_filtered = _apply_fft_frequency_mask(fshift , rows , cols)

    # ifftshift reverses the quadrant swap before the inverse transform so that
    # the array layout matches what np.fft.ifft2 expects.
    f_ishift = np.fft.ifftshift(fshift_filtered)

    # Inverse FFT: reconstruct the spatial image from the filtered spectrum.
    # np.abs discards the tiny imaginary residuals caused by floating-point
    # rounding — the result should be real for a real-input FFT.
    img_back = np.fft.ifft2(f_ishift)
    cleaned = np.abs(img_back)

    # Normalise to float32 [0, 1] for downstream blending without uint8 clamping.
    cleaned = cv2.normalize(cleaned, None, 0, 1.0,
                            cv2.NORM_MINMAX).astype(np.float32)

    return cleaned

def _read_gray(path:str ):
    """
    Read an image file as a float32 grayscale array in [0, 1].

    Parameters
    ----------
    path : str
        File-system path to the image.

    Returns
    -------
    numpy.ndarray, dtype=float32, shape=(H, W)
        Grayscale pixel values normalised to [0.0, 1.0].

    Raises
    ------
    FileNotFoundError
        If ``cv2.imread`` cannot open the file at ``path``.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Could not read image : {path}")
    # Divide by 255 to map [0, 255] uint8 → [0.0, 1.0] float32.
    # clip(0, 1) guards against rare uint8 overflow artefacts.
    return (img.astype(np.float32) / 255.0).clip(0 , 1)

def debug_apply_ref_watermark():
    pass

def _get_distance_maps(msg01: np.ndarray):
    """
    Compute Euclidean distance transforms for the watermark mask.

    ``cv2.distanceTransform`` computes, for every pixel, the distance to the
    nearest pixel of the *opposite* label (foreground vs. background).  We
    need two such maps:

    • **di** — distance of each *foreground* (ink) pixel to the nearest
      background pixel.  Used to measure how deep inside a stroke a pixel sits.
    • **do** — distance of each *background* (non-ink) pixel to the nearest
      foreground pixel.  Used to measure how far outside a stroke a pixel sits.

    These maps enable smooth, distance-based falloff for the glow and shadow
    effects, avoiding the hard binary edge of a simple threshold mask.

    Parameters
    ----------
    msg01 : numpy.ndarray, dtype=float32, values in [0, 1]
        Grayscale watermark image (0 = ink / dark, 1 = background / bright)
        normalised to [0, 1].

    Returns
    -------
    hard : numpy.ndarray, dtype=uint8
        Binary mask: 1 = ink (foreground), 0 = background; derived by Otsu
        thresholding on the *inverted* image so dark pixels become foreground.
    di : numpy.ndarray, dtype=float32
        For each foreground pixel: Euclidean distance (in pixels) to the
        nearest background pixel.
    do : numpy.ndarray, dtype=float32
        For each background pixel: Euclidean distance (in pixels) to the
        nearest foreground pixel.
    """
    # Invert so that dark ink pixels (near 0 in msg01) become bright (near 1)
    # before thresholding, making them the foreground for distanceTransform.
    inv = (1.0 - msg01)
    inv_u8 = (np.clip(inv , 0 , 1) * 255).astype(np.uint8)

    # Otsu's method automatically finds the optimal global threshold that
    # separates the bimodal (ink vs. background) histogram.
    _ , hard_u8 = cv2.threshold(inv_u8 , 0 , 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hard = hard_u8.astype(np.uint8)

    # cv2.distanceTransform(src, distanceType, maskSize):
    #   • DIST_L2 computes the true Euclidean (L2) distance.
    #   • maskSize=3 uses a 3×3 approximation kernel; sufficient for sub-pixel
    #     accuracy at the pixel distances we work with (< ~30 px).
    # di: distance from each ink pixel to the nearest background pixel.
    di = cv2.distanceTransform(hard, cv2.DIST_L2, 3).astype(np.float32)
    # do: distance from each background pixel to the nearest ink pixel.
    # We flip 'hard' (1−hard) so the roles of foreground and background swap.
    do = cv2.distanceTransform(1 - hard, cv2.DIST_L2, 3).astype(np.float32)

    return hard , di , do

def _get_regions(hard: np.ndarray , di: np.ndarray, do:np.ndarray , radius_px: float, inner_width_px:float):
    """
    Identify the outer glow band and inner shadow band around watermark strokes.

    The two regions are defined purely by distance constraints:

    • **outer_region** — background pixels (hard == 0) whose distance to the
      nearest ink stroke (``do``) is within ``radius_px``.  These pixels will
      receive the outer glow brightening effect.
    • **inner_region** — ink pixels (hard == 1) whose distance to the nearest
      ink–background boundary (``di``) is within ``inner_width_px``.  These
      pixels will receive the inner shadow darkening effect.

    Parameters
    ----------
    hard : numpy.ndarray, dtype=uint8
        Binary ink mask (1 = ink, 0 = background).
    di : numpy.ndarray, dtype=float32
        Distance map for foreground pixels (see ``_get_distance_maps``).
    do : numpy.ndarray, dtype=float32
        Distance map for background pixels (see ``_get_distance_maps``).
    radius_px : float
        Maximum distance (pixels) from a stroke at which the outer glow
        is still applied.
    inner_width_px : float
        Maximum distance (pixels) from the stroke boundary at which the
        inner shadow is still applied.

    Returns
    -------
    outer_region : numpy.ndarray, dtype=bool
        Boolean mask for the outer glow band.
    inner_region : numpy.ndarray, dtype=bool
        Boolean mask for the inner shadow band.
    """
    # Outer glow: background pixels close enough to an ink stroke.
    # do > 0 excludes the exact ink boundary pixel itself.
    outer_region = (hard == 0) & (do > 0) & (do <= radius_px)
    # Inner shadow: ink pixels close to the stroke's outer edge.
    # di > 0 excludes the exact boundary pixel.
    inner_region = (hard == 1) & (di > 0) & (di <= inner_width_px)

    return outer_region , inner_region

def _feather_local(w: np.ndarray, sigma: float, region: np.ndarray):
    """
    Apply a localised Gaussian blur to a weight map, confined to a region mask.

    After computing a distance-based falloff, the weight map ``w`` can still
    have subtle quantisation banding because distance values are discrete at
    integer pixel locations.  A light Gaussian blur smooths these bands.

    **Why localised?**  Gaussian blur has non-zero support (it spreads values
    across nearby pixels).  Without masking, blur would leak glow/shadow weights
    into pixels outside the intended region.  By zeroing out ``ww`` wherever
    ``region`` is False, we ensure the effect stays strictly within its band.

    Parameters
    ----------
    w : numpy.ndarray, dtype=float32, shape=(H, W)
        Weight map to feather.  Pixels outside ``region`` should already be 0.
    sigma : float
        Gaussian standard deviation in pixels.  Larger values → smoother falloff.
    region : numpy.ndarray, dtype=bool, shape=(H, W)
        The active pixels; blur is applied globally but then masked to this region.

    Returns
    -------
    numpy.ndarray, dtype=float32, shape=(H, W)
        Feathered weight map, zero outside ``region``.
    """
    # Early exit: if the entire weight map is zero there is nothing to blur.
    if w.max() <= 0:
        return w

    # Kernel size: at least 3, rounded up to the next odd integer.
    # GaussianBlur requires an odd kernel size; sigma*3 covers ≈ 99.7 % of
    # the Gaussian's energy, so the kernel is large enough to be accurate.
    k = max(3 , int(sigma * 3) | 1)

    ww = cv2.GaussianBlur(w, (k ,k) , sigma)
    # Re-mask: zero out any blurred values that leaked outside the active region.
    ww[~region] = 0.0

    return ww

def _compute_intensities(H: int, W: int, do: np.ndarray, di: np.ndarray,
                         outer_region, inner_region, radius_px: float,
                         falloff_power: float, inner_width_px:float, inner_power: float):
    """
    Compute per-pixel intensity weights for the outer glow and inner shadow.

    The weight at each active pixel is a **power-law falloff** of normalised
    distance from the stroke boundary::

        glow[px]   = (1 − do[px] / radius_px)   ^ falloff_power
        shadow[px] = (1 − di[px] / inner_width_px) ^ inner_power

    This produces a smooth gradient that is 1.0 at the stroke edge and decays
    to 0.0 at the band boundary, with the exponent controlling the steepness
    of the falloff curve (higher power → sharper, more concentrated effect).

    Both maps are subsequently feathered with ``_feather_local`` to remove
    residual quantisation banding from the integer-resolution distance maps.

    Parameters
    ----------
    H, W : int
        Image height and width in pixels.
    do : numpy.ndarray, dtype=float32
        Outer distance map (background pixels → nearest ink stroke).
    di : numpy.ndarray, dtype=float32
        Inner distance map (ink pixels → nearest ink boundary).
    outer_region : numpy.ndarray, dtype=bool
        Mask of pixels eligible for the outer glow effect.
    inner_region : numpy.ndarray, dtype=bool
        Mask of pixels eligible for the inner shadow effect.
    radius_px : float
        Maximum reach of the outer glow in pixels.
    falloff_power : float
        Exponent of the outer glow power-law decay; higher → sharper falloff.
    inner_width_px : float
        Maximum reach of the inner shadow in pixels.
    inner_power : float
        Exponent of the inner shadow power-law decay.

    Returns
    -------
    glow : numpy.ndarray, dtype=float32, shape=(H, W)
        Outer glow weight map (0.0–1.0), feathered and masked to outer_region.
    inn : numpy.ndarray, dtype=float32, shape=(H, W)
        Inner shadow weight map (0.0–1.0), feathered and masked to inner_region.
    """
    glow = np.zeros((H , W), np.float32)
    inn = np.zeros((H , W), np.float32)

    if np.any(outer_region):
        R = float(radius_px)
        p = float(falloff_power)
        # Normalise distance to [0, 1] then apply power-law: 1 at stroke edge,
        # 0 at the outer boundary.  np.clip guards against floating-point noise.
        glow[outer_region] = np.clip(
            1.0 - (do[outer_region] / R) , 0.0 , 1.0) ** p

    if np.any(inner_region):
        ri = float(radius_px)
        q = float(falloff_power)
        # Same power-law for the inner shadow using its own distance / width.
        inn[inner_region] = np.clip(
            1.0 - (di[inner_region] / ri) , 0.0 , 1.0) ** q

    # Feather the glow weight map with a narrow Gaussian to smooth banding.
    # sigma=0.7 → kernel size = max(3, int(0.7*3)|1) = 3, very light touch.
    glow = _feather_local(glow , 0.7 , outer_region)
    # Slightly tighter feathering for the inner shadow (sigma=0.6).
    inn = _feather_local(inn , 0.6, inner_region)

    return glow , inn

def _blend_effects_to_base(base: np.ndarray, glow: np.ndarray, inn: np.ndarray,
                           outer_region, inner_region,  glow_strength:float , inner_strength: float):
    """
    Additively blend the outer glow and inner shadow onto the base image.

    **Outer glow** (additive brightening)::

        out[px] = base[px] + glow_strength × glow[px] × (1 − base[px])

    The factor ``(1 − base[px])`` limits the glow on already-bright pixels
    (avoids over-exposure / clipping), giving a physically plausible behaviour
    where glow has less effect in bright areas.

    **Inner shadow** (subtractive darkening)::

        out[px] = base[px] − inner_strength × inn[px] × (1 − base[px])

    This darkens the ink-edge pixels relative to the base, producing a
    recessed / embossed shadow effect.  The same (1 − base) term ensures the
    darkening stays proportional to available headroom.

    Parameters
    ----------
    base : numpy.ndarray, dtype=float32, values in [0, 1]
        De-banded cover image.
    glow : numpy.ndarray, dtype=float32, values in [0, 1]
        Outer glow weight map from ``_compute_intensities``.
    inn : numpy.ndarray, dtype=float32, values in [0, 1]
        Inner shadow weight map from ``_compute_intensities``.
    outer_region : numpy.ndarray, dtype=bool
        Mask selecting outer glow pixels.
    inner_region : numpy.ndarray, dtype=bool
        Mask selecting inner shadow pixels.
    glow_strength : float
        Overall multiplier for the glow effect; range [0, 1].
    inner_strength : float
        Overall multiplier for the shadow effect; range [0, 1].

    Returns
    -------
    numpy.ndarray, dtype=float32, values in [0, 1]
        Blended image, clipped to [0, 1].
    """
    out = base.copy()

    # Additive outer glow: brightens background pixels near the stroke.
    out[outer_region] = out[outer_region] + glow_strength * glow[outer_region] * (1.0 - out[outer_region])

    # Subtractive inner shadow: darkens ink-edge pixels for the embossed look.
    out[inner_region] = out[inner_region] - inner_strength * inn[inner_region] * (1.0 - out[inner_region])

    # Clip to valid floating-point image range to prevent display artefacts.
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def _apply_ref_watermark(base01: np.ndarray,
                         msg01: np.ndarray,
                         radius_px: float = 4.0,
                         glow_strength: float = 0.95,
                         falloff_power: float = 1.5,
                         inner_width_px: float = 2.8,
                         inner_strength: float = 0.95,
                         inner_power: float = 1.8):
    """
    Orchestrate the full distance-transform glow watermark effect.

    This is the main algorithm function.  It calls the helper chain:
    ``_get_distance_maps`` → ``_get_regions`` → ``_compute_intensities``
    → ``_blend_effects_to_base`` and returns the composited image.

    Parameters
    ----------
    base01 : numpy.ndarray, dtype=float32, shape=(H, W), values in [0, 1]
        De-banded grayscale cover image.
    msg01 : numpy.ndarray, dtype=float32, shape=(H, W), values in [0, 1]
        Grayscale watermark image (0 = dark ink, 1 = bright background).
    radius_px : float, optional
        Outer glow reach in pixels.  Default 4.0.
    glow_strength : float, optional
        Peak brightness boost for the outer glow.  Default 0.95 (very strong).
    falloff_power : float, optional
        Power-law exponent controlling outer glow decay speed.  Default 1.5.
    inner_width_px : float, optional
        Inner shadow reach in pixels.  Default 2.8.
    inner_strength : float, optional
        Peak darkening for the inner shadow.  Default 0.95 (very strong).
    inner_power : float, optional
        Power-law exponent controlling inner shadow decay speed.  Default 1.8.

    Returns
    -------
    numpy.ndarray, dtype=float32, values in [0, 1]
        Composited image with glow watermark applied.
    """
    base = base01.astype(np.float32)
    H , W = base.shape

    # Step 1: Threshold the watermark and compute inner/outer distance fields.
    hard , di , do = _get_distance_maps(msg01)

    # Step 2: Determine which pixels fall in the outer glow and inner shadow bands.
    outer_region , inner_region = _get_regions(
        hard , di, do, radius_px, inner_width_px
    )

    # Step 3: Compute power-law intensity weights + feather them.
    glow , inn = _compute_intensities(
        H , W , do , di, outer_region , inner_region,
        radius_px , falloff_power , inner_width_px, inner_power
    )

    # Step 4: Additively blend glow and shadow onto the base image.
    out = _blend_effects_to_base(
        base , glow , inn , outer_region , inner_region,
        glow_strength , inner_strength
    )

    return out

def _to_uint8(x: np.ndarray):
    """
    Convert a float32 image in [0, 1] to uint8 [0, 255].

    Adding 0.5 before casting achieves correct rounding (Python's int()
    truncates, so without +0.5 a value of 0.9999 would map to 254 not 255).

    Parameters
    ----------
    x : numpy.ndarray, dtype=float32
        Image with values in [0.0, 1.0].

    Returns
    -------
    numpy.ndarray, dtype=uint8
        Image with values in [0, 255].
    """
    return (np.clip(x , 0 , 1) * 255.0 + 0.5).astype(np.uint8)

def apply_watermark(cover_name: str, message_name:str):
    """
    Public entry point: read cover and watermark files, run the full pipeline.

    Executes the complete watermarking workflow:
    1. Reads the cover and watermark images as float32 grayscale [0, 1].
    2. Removes vertical banding from the cover via FFT filtering.
    3. Applies the distance-transform glow watermark effect.
    4. Returns the composited result as a uint8 image.

    Parameters
    ----------
    cover_name : str
        Basename of the cover image file (looked up in ``Data/Cover/``).
    message_name : str
        Basename of the watermark image file (looked up in ``Data/Message/``).

    Returns
    -------
    numpy.ndarray, dtype=uint8, shape=(H, W)
        Watermarked grayscale image ready for display or saving.
    """
    cover_path = os.path.join("data/cover", cover_name)
    msg_path = os.path.join("data/message", message_name)

    # Read both images as float32 [0, 1] grayscale.
    cover = _read_gray(cover_path)
    msg = _read_gray(msg_path)

    # Denoise the cover before watermarking to avoid banding artefacts
    # interacting with the glow effect.
    debanded = remove_vertical_noise(cover)

    # Apply the distance-transform glow watermark and get result in [0, 1].
    out01 = _apply_ref_watermark(debanded , msg)

    # Convert to uint8 for output / display.
    return _to_uint8(out01)