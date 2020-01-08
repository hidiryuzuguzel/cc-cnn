import cv2
import numpy as np


def resize_image(img, maxwh = 1200):
    """
    Resize images s.t. max(w,h) = 1200

    Args:
        img (np.array): Input image to be resized
        maxwh (int):    Number of pixels for the longest dimension
    Returns:
        img_resized (np.array): output resized image
    """

    r, c, _ = img.shape
    img_scale = maxwh/np.maximum(r, c)
    img_resized = cv2.resize(img, (int(np.ceil(c*img_scale)), int(np.ceil(r*img_scale))))
    return img_resized

def hist_stretch(img, bit_depth):
    """
    Contrast normalization via global histogram stretching

    Args:
        img (np.array): Input image
        bit_depth (int): Number of bits per channel
    Returns:
        img_out (np.array): Contast normalized image
    """

    if np.max(img) - np.min(img) < 1e-5: # Do-nothing
        img_out = img
    else:
        min_img, max_img = np.min(img), np.max(img)
        img_out = ((img - min_img) / (max_img - min_img)) * (2 ** bit_depth - 1)

    return img_out

def srgb_gamma(img):
    """
    Apply gamma correction (forward transformation)
    (https://en.wikipedia.org/wiki/SRGB)

    Args:
        img (np.array): input (linear) image
    Returns:
        img (np.array): non-linear/gamma-compressed image
    """

    for idx_channel in range(img.shape[2]):
        this_channel = img[:, :, idx_channel]
        img[:, :, idx_channel] = 12.92 * this_channel * (this_channel <= 0.0031308) + (
                    1.055 * this_channel ** (1 / 2.4) - 0.055) * (this_channel > 0.0031308)

    return img


def correct_color_single_image(img, illuminant):
    """
    Apply color correction via Von-Kries Model

    Args:
        img (np.array): input (linear) image
        illuminant (np.array): RGB  color of light source
    Returns:
        img_corr (np.array): corrected image s.t. to be
        taken under a canonical perfect white light source
    """
    highest_gain = np.max(illuminant)
    gain = highest_gain / illuminant
    img_corr = img.copy()
    for idx_channel in range(img.shape[2]):
        img_corr[:, :, idx_channel] = img_corr[:, :, idx_channel] * gain[idx_channel]

    return img_corr