import numpy as np
import cv2
import configparser
from imaging import hist_stretch, resize_image
from utils import get_images_fullpath, load_groundtruth_illuminant

def roipoly(img, c, r):
    """
    Specify polygonal region of interest (ROI).

    Args:
        img (np.array): Input image
        c (list): x-coordinate of polygon vertices
        r (list): y-coordinate of polygon vertices
    Returns:
        mask (np.array): binary mask
    """

    mask = np.zeros(img.shape[:2], dtype='uint8')
    rc = np.column_stack((c,r))
    cv2.drawContours(mask, [rc.astype(int)], 0, 255, -1)
    return mask.astype(bool)


def generate_patch_corner_positions(row, col, patch_size):
    """
    Generate patch positions for an image specified by
    number of row and column

    Args:
        row (int): image row number
        col (int): image column number
        patch_size (int): patch size
    Returns:
        patch_corners (list): Upper left position (corner) of the patches
    """

    X = np.arange(0, row, patch_size)
    Y = np.arange(0, col, patch_size)

    # if the last patch goes beyond the boundary, remove it
    if X[-1] + patch_size >= row:
        X = X[:-1]
    if Y[-1] + patch_size >= col:
        Y = Y[:-1]

    xv, yv = np.meshgrid(X, Y)
    patch_corners = np.column_stack((xv.flatten(), yv.flatten()))
    return patch_corners

def remove_patches_overlapping_mcc(patch_corners, mcc_mask, patch_size, max_overlapping=0):
    """
    Remove patches that overlap with the macbeth color checker (mcc)

    Args:
        patch_corners (list): Upper left position of the patches
        mcc_mask (np.array): mask array of macbeth color checker
        patch_size (int): patch size
        max_overlapping (float): upper limit of overlapping percentage with the mcc
    Returns:
        filtered_patch_corners (list): Filtered upper left position (corner) of the patches
    """

    filtered_patch_corners = []
    for patch_corner in patch_corners:
        r, c = patch_corner
        if np.sum(mcc_mask[r:r + patch_size, c:c + patch_size]) <= max_overlapping * patch_size * patch_size:
            filtered_patch_corners.append(patch_corner)
    filtered_patch_corners = np.asarray(filtered_patch_corners)
    return filtered_patch_corners

def select_most_brightest_patches(img, patch_corners, npatches_per_image, patch_size):
    """
    Take the most brightest patches from the image. The brightness is defined as
    the sum of all pixel intensities of RGB channels in the patch

    Args:
        img (np.array): Input image
        patch_corners (list): Upper left position of the patches
        npatches_per_image (int): Number of patches to be extracted per image
        patch_size (int): patch size
    Returns:
        filtered_patch_corners (list): Filtered upper left position (corner) of the patches
    """

    brightness = []
    for patch_corner in patch_corners:
        r, c = patch_corner
        img_patch = img[r:r + patch_size, c:c + patch_size, :]
        img_patch_brightness = np.sum(img_patch)
        brightness.append(img_patch_brightness)
    brightness = np.asarray(brightness)
    ids = np.argsort(brightness)[::-1] # due to the descending order
    filtered_patch_corners = patch_corners[ids[:npatches_per_image], :]
    return filtered_patch_corners


def normalize_patches(patches):
    """
    Zero mean & unit variance feature standardization

    Args:
        patches (np.array): Input patches
    Returns:
        img_out (np.array): Contast normalized image
    """

    from sklearn import preprocessing

    npatches, patch_size, _, nchannels = np.shape(patches)
    patches = np.reshape(patches, (npatches, patch_size * patch_size * nchannels))
    normalized_patches = preprocessing.scale(patches)
    normalized_patches = np.reshape(normalized_patches, (npatches, patch_size, patch_size, nchannels))
    return normalized_patches

def get_shigehler_patch_corners(shigehler_config):
    """
    Wrapper function to compute corners of patches for ShiGehler dataset

    Args:
        shigehler_config (cfg): config for shigehler dataset
    Returns:
        shigehler_patch_corners (np.array): Corners of patches
    """
    print("Getting Shigehler patch corners...")

    img_folder_path = shigehler_config["SHIGEHLER"]["DB_PATH"]
    mcc_coordinate_path = shigehler_config["SHIGEHLER"]["MCC_MASK_COORDINATE_PATH"]
    cc_file_extension = shigehler_config["SHIGEHLER"]["MCC_FILE_EXTENSION"]
    remove_mask = int(shigehler_config["SHIGEHLER"]["REMOVE_MASK"])
    patch_size = int(shigehler_config["SHIGEHLER"]["PATCH_SIZE"])
    npatches_per_image = int(shigehler_config["SHIGEHLER"]["NPATCHES_PER_IMAGE"])

    images_fullpath = get_images_fullpath(img_folder_path)

    shigehler_patch_corners = []

    for idx_image, img_fullpath in enumerate(images_fullpath):

        img = cv2.imread(img_fullpath, cv2.IMREAD_UNCHANGED).astype(np.float64)
        img = img[:, :, ::-1]  # convert BGR to RGB
        # subtract black level: For the Canon 5D the black level is 129 and
        # for the Canon 1D it is zero.
        if "canon5d" in img_fullpath:
            img = np.maximum(img - 129, 0)

        img = resize_image(img)
        nrow, ncol, _ = img.shape
        img_name = img_fullpath.split('/')[-1].split('.')[0]

        # mask out the colorchecker within the image scene
        cc_coord = np.loadtxt(mcc_coordinate_path + img_name + cc_file_extension)
        scale = cc_coord[0][[1, 0]] / np.array([nrow, ncol])
        if remove_mask:
            mask = roipoly(img, cc_coord[[2, 4, 5, 3], 0] / scale[0], cc_coord[[2, 4, 5, 3], 1] / scale[1])
        else:
            mask = np.zeros(img.shape[:2])
        patch_corners = generate_patch_corner_positions(nrow, ncol, patch_size)
        # don't take patches from the macbeth color checker
        patch_corners = remove_patches_overlapping_mcc(patch_corners, mask, patch_size)
        # select the most brightest patches
        patch_corners = select_most_brightest_patches(img, patch_corners, npatches_per_image, patch_size)
        image_idx_and_grid = np.column_stack([idx_image * np.ones((patch_corners.shape[0], 1)), patch_corners])
        shigehler_patch_corners.append(image_idx_and_grid)

    shigehler_patch_corners = np.concatenate(shigehler_patch_corners)

    return shigehler_patch_corners

def extract_shigehler_patches(shigehler_patch_corners, shigehler_config):
    """
    Wrapper function to extract patches from ShiGehler dataset

    Args:
        shigehler_patch_corners (np.array): Corners of patches
        shigehler_config (cfg): config for shigehler dataset
    Returns:
        patches (np.array): patches
        patch_labels (np.array): ground truth illuminant values
        patch_to_image (np.array): image idx of patches
    """

    print("Extracting Shigehler patches...")

    img_folder_path = shigehler_config["SHIGEHLER"]["DB_PATH"]
    patch_size = int(shigehler_config["SHIGEHLER"]["PATCH_SIZE"])
    npatches_per_image = int(shigehler_config["SHIGEHLER"]["NPATCHES_PER_IMAGE"])
    nimage_channels = int(shigehler_config["SHIGEHLER"]["NIMAGECHANNELS"])
    original_bitdepth = int(shigehler_config["SHIGEHLER"]["ORIGINAL_BITDEPTH"])
    gt_file_path = shigehler_config["SHIGEHLER"]["REAL_RGB"]

    images_fullpath = get_images_fullpath(img_folder_path)
    real_rgb = load_groundtruth_illuminant(gt_file_path)

    npatches = len(images_fullpath) * npatches_per_image
    patches = np.zeros((npatches, patch_size, patch_size, nimage_channels))
    patch_labels = np.zeros((npatches, 3))
    patch_to_image = np.zeros(npatches)

    for idx_image, img_fullpath in enumerate(images_fullpath):
        img = cv2.imread(img_fullpath, cv2.IMREAD_UNCHANGED).astype(np.float64)
        img = img[:,:,::-1] # convert BGR to RGB
        img = resize_image(img)
        if "canon5d" in img_fullpath:  # subtract black level
            img = np.maximum(img - 129, 0)

        img_patch_idx = np.where(shigehler_patch_corners[:, 0] == idx_image)[0]
        for p in range(npatches_per_image):
            idx_patch = (idx_image * npatches_per_image) + p
            sr = int(shigehler_patch_corners[img_patch_idx[p], 1])  # starting row
            sc = int(shigehler_patch_corners[img_patch_idx[p], 2])  # starting col
            img_patch = img[sr:sr + patch_size, sc:sc + patch_size, :]
            img_patch = hist_stretch(img_patch, original_bitdepth)
            patches[idx_patch, ...] = img_patch
            patch_labels[idx_patch, :] = real_rgb[idx_image, :]
            patch_to_image[idx_patch] = idx_image

    return patches, patch_labels, patch_to_image

def main(shigehler_config):

    shigehler_patch_corners = get_shigehler_patch_corners(shigehler_config)
    patches, patch_labels, image_idx = extract_shigehler_patches(shigehler_patch_corners, shigehler_config)
    data = normalize_patches(patches)
    np.savez('./assets/shigehler.npz', data=data, patch_labels=patch_labels, image_idx=image_idx)

if __name__ == '__main__':

    shigehler_config = configparser.ConfigParser()
    shigehler_config.read("shigehler.cfg")

    main(shigehler_config)