import glob
import numpy as np
from scipy.io import loadmat

def get_images_fullpath(img_folder_path):
    """ get all the images for the specified folder

    Args:
        img_folder_path (str): path which contains the images
        img_file_format (str): image file format
    Returns:
        images_fullpath (list):  list of image (full) paths
    """

    images_fullpath = glob.glob(img_folder_path + "**/*.png", recursive=True)
    images_fullpath.sort(key=lambda x: x.split('/')[-1].split('.')[0])
    return images_fullpath

def load_groundtruth_illuminant(file_path):
    """ load ground truth illuminant

    Args:
        file_path (str): path which contains the ground truth illuminant values
    Returns:
        real_rgb (np.array):  ground truth illuminant values
    """

    real_illum_568 = loadmat(file_path)
    real_rgb = real_illum_568["real_rgb"]
    real_rgb = real_rgb / real_rgb[:, 1][:, np.newaxis]  # convert to chromaticity
    return real_rgb