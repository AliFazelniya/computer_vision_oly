import cv2
import numpy as np


def remove_vertical_noise(img):

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    fshift_filtered = fshift.copy()

    mask_height = 1
    mask_width = max(3, cols // 100)

    fshift_filtered[crow - mask_height:crow +
                    mask_height, :ccol - mask_width] = 0
    fshift_filtered[crow - mask_height:crow +
                    mask_height, ccol + mask_width:] = 0

    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    cleaned = np.abs(img_back)

    cleaned = cv2.normalize(cleaned, None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)

    if len(img.shape) == 3:
        cleaned = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    return cleaned


def apply_watermark(cover_name, message_name):

    source_img = cv2.imread(cover_name, 0)
    watermark = cv2.imread(message_name, 0)

    cleaned_img = remove_vertical_noise(source_img)
    watermark = cv2.resize(
        watermark, (cleaned_img.shape[1], cleaned_img.shape[0]))
    _ , mask = cv2.threshold(watermark, 200, 255, cv2.THRESH_BINARY_INV)

    result = cleaned_img.copy().astype(np.float32)

    alpha = 0.3 

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
