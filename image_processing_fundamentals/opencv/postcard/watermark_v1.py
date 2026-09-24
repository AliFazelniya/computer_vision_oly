"""
watermark_v1.py — FFT Noise Removal + Alpha-Blend Watermarking (v1)
====================================================================

This module provides a two-stage pipeline for embedding a visible watermark
onto a postcard image:

Stage 1 — Vertical Banding Noise Removal (FFT)
-----------------------------------------------
Postcard scans often exhibit regular vertical banding artefacts caused by
scanner CCD-column variation or mains hum interference.  These appear in the
frequency domain as bright spots aligned along the horizontal axis of the
spectrum (i.e., they have zero or very low vertical frequency).  The function
``remove_vertical_noise`` isolates those frequency components and zeros them
out, then reconstructs the cleaned image via the inverse FFT.

Stage 2 — Alpha-Blend Watermarking
------------------------------------
The function ``apply_watermark`` reads the watermark image, thresholds it to
obtain a binary pixel mask (ink vs. background), and additively darkens the
cover image at watermark ink locations using a weighted blend.  This is the
simplest form of spatial-domain watermarking and is deliberately lightweight
for easy comprehension.

Typical usage
-------------
    python watermark_v1.py

    The script reads ./Data/Cover/Kashan.jpg and ./Data/Message/Kashan.jpg,
    applies both stages, and displays the result.

See Also
--------
watermark_v2.py — refined version with distance-transform-based glow effects.
"""

import cv2
import numpy as np


def remove_vertical_noise(img):
    """
    Remove vertical banding artefacts from an image using 2-D FFT filtering.

    **How it works — frequency-domain intuition**

    A 2-D Fast Fourier Transform (FFT) decomposes the spatial image into a
    superposition of sinusoidal frequency components.  After applying
    ``np.fft.fftshift``, low-frequency (smooth / global) components sit near
    the centre of the spectrum array, while high-frequency (sharp / fine)
    components sit near the edges.

    Vertical banding (stripes of constant width running top-to-bottom) has a
    characteristic spectral signature: it produces bright peaks along the
    *horizontal* centre row of the shifted spectrum (because the banding
    repeats across columns, its spatial frequency vector points in the
    horizontal direction, i.e., has zero vertical component).

    By zeroing out a narrow horizontal slab around the spectrum centre row
    (while leaving a small protected zone at the exact DC centre to avoid
    corrupting the image mean), we surgically remove only those frequency
    components responsible for the banding, leaving all other spatial
    frequencies intact.

    Parameters
    ----------
    img : numpy.ndarray
        Input image (BGR colour or single-channel grayscale, uint8).

    Returns
    -------
    numpy.ndarray
        De-banded image.  If the input was BGR colour, the output is BGR.
        If the input was grayscale, the output is grayscale.

    Notes
    -----
    • The FFT operates on the grayscale (luminance) channel only; the colour
      image is converted to grey for filtering and then back to BGR so that
      chroma information is preserved (or rather, reconstructed from the
      cleaned luma — a deliberate simplification for this exercise).
    • ``mask_width`` is proportional to the image width (cols // 100) so the
      protected DC zone scales with resolution.
    """
    # Convert to grayscale if the input is a colour image.
    # FFT-based frequency analysis is performed on a single channel for simplicity.
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # --- Forward FFT ---
    # np.fft.fft2 computes the 2-D Discrete Fourier Transform of the image,
    # returning a complex-valued array F(u, v) where each element encodes the
    # amplitude and phase of a spatial sinusoid at frequency (u, v).
    f = np.fft.fft2(gray)

    # np.fft.fftshift rearranges the quadrants of the spectrum so that the DC
    # component (zero frequency, i.e., the image mean) moves to the array centre.
    # This makes the spectrum visually intuitive: low frequencies are central,
    # high frequencies are peripheral.
    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2  # Centre pixel of the shifted spectrum

    fshift_filtered = fshift.copy()

    # --- Define the banding suppression mask ---
    # mask_height = 1 means we suppress a 2-pixel-tall horizontal stripe
    # (rows crow-1 to crow+1) in the shifted spectrum.
    # mask_width protects a small neighbourhood around the DC column so we
    # do not remove the global image mean, which would cause a massive brightness
    # shift in the reconstructed image.
    mask_height = 1
    mask_width = max(3, cols // 100)

    # Zero out the horizontal slab LEFT of the DC column.
    # These complex coefficients correspond to leftward-moving sinusoids that
    # contribute to the horizontal banding pattern.
    fshift_filtered[crow - mask_height:crow +
                    mask_height, :ccol - mask_width] = 0

    # Zero out the horizontal slab RIGHT of the DC column (symmetric partner).
    # The FFT spectrum is conjugate-symmetric for real input, so both halves
    # must be zeroed to ensure the reconstructed image remains real-valued.
    fshift_filtered[crow - mask_height:crow +
                    mask_height, ccol + mask_width:] = 0

    # --- Inverse FFT ---
    # np.fft.ifftshift undoes the quadrant rearrangement before the inverse
    # transform so that the array layout matches what ifft2 expects.
    f_ishift = np.fft.ifftshift(fshift_filtered)

    # np.fft.ifft2 reconstructs the spatial image from the filtered spectrum.
    # Because we zeroed some coefficients the result is still complex (tiny
    # imaginary residuals from floating-point rounding); np.abs discards them.
    img_back = np.fft.ifft2(f_ishift)
    cleaned = np.abs(img_back)

    # Normalise the reconstructed float values back to [0, 255] uint8.
    # Frequency-domain modifications can shift the dynamic range slightly.
    cleaned = cv2.normalize(cleaned, None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)

    # Restore colour format if the original input was BGR.
    if len(img.shape) == 3:
        cleaned = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    return cleaned


def apply_watermark(cover_name, message_name):
    """
    Embed a visible watermark onto a cover image using alpha blending.

    The pipeline:
    1. Read both images as grayscale.
    2. Remove vertical banding noise from the cover (see ``remove_vertical_noise``).
    3. Resize the watermark to match the cover dimensions.
    4. Threshold the watermark to isolate ink pixels (dark regions → mask).
    5. Darken the cover at ink-pixel locations by the alpha factor.

    **Alpha-blending rationale**

    For each ink pixel p in the cover::

        result[p] = cover[p] × (1 − alpha)

    This reduces the cover brightness at watermark locations, making the
    watermark text/logo appear as a dark impression.  alpha = 0.3 means the
    cover retains 70 % of its original brightness where the watermark is
    present, producing a semi-transparent effect.

    A binary mask is used rather than blending with the watermark pixel values
    directly because the watermark image may contain gradients or JPEG
    compression artefacts; thresholding produces a clean, deterministic result.

    Parameters
    ----------
    cover_name : str
        Path to the cover (background) image file.
    message_name : str
        Path to the watermark (message) image file.

    Returns
    -------
    numpy.ndarray, dtype=uint8
        Watermarked image with the same spatial dimensions as the cover.
    """
    # Read both images in grayscale mode (flag 0 = cv2.IMREAD_GRAYSCALE).
    source_img = cv2.imread(cover_name, 0)
    watermark = cv2.imread(message_name, 0)

    # Denoise the cover before watermarking so banding artefacts do not
    # interfere with the final visual quality.
    cleaned_img = remove_vertical_noise(source_img)

    # Scale the watermark to the cleaned cover's spatial dimensions.
    # Default interpolation (INTER_LINEAR) is fine here since the watermark
    # will be thresholded immediately after.
    watermark = cv2.resize(
        watermark, (cleaned_img.shape[1], cleaned_img.shape[0]))

    # cv2.threshold with THRESH_BINARY_INV + threshold value 200:
    #   • pixels with value > 200 (near-white / background) → 0  (not ink)
    #   • pixels with value ≤ 200 (dark ink strokes)        → 255 (ink)
    # This converts the greyscale watermark into a clean binary ink mask.
    _ , mask = cv2.threshold(watermark, 200, 255, cv2.THRESH_BINARY_INV)

    # Work in float32 to avoid integer overflow / truncation during blending.
    result = cleaned_img.copy().astype(np.float32)

    # Alpha-blend: darken the cover at ink pixel locations.
    # alpha = 0.3 → watermark visibility is 30 % of original brightness delta.
    # Increasing alpha makes the watermark darker (more opaque).
    alpha = 0.3 

    # Apply the blend only to the ink-pixel locations identified by the mask.
    # Non-ink pixels are left unchanged, preserving the cover image fidelity.
    result[mask == 255] = result[mask == 255] * (1 - alpha)

    return result.astype(np.uint8)


def main():
    cover_name = "./Data/Cover/Kashan.jpg"
    message_name = "./Data/Message/Kashan.jpg"

    result = apply_watermark(cover_name, message_name)
    cv2.imshow("result", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
