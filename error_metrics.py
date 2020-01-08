import numpy as np

def compute_angular_error(y_true, y_pred):
    """
    Angle between the RGB triplet of the measured ground truth
    illumination and RGB triplet of estimated illuminant

    Args:
        y_true (np.array): ground truth RGB illuminants
        y_pred (np.array): predicted RGB illuminants
    Returns:
        err (np.array):  angular error
    """

    gt_norm = np.linalg.norm(y_true, axis=1)
    gt_normalized = y_true / gt_norm[..., np.newaxis]
    est_norm = np.linalg.norm(y_pred, axis=1)
    est_normalized = y_pred / est_norm[..., np.newaxis]
    dot = np.sum(gt_normalized * est_normalized, axis=1)
    err = np.degrees(np.arccos(dot))
    return err

def compute_angular_error_stats(ang_err):
    """
    Angular error statistics such as min, max, mean, etc.

    Args:
        ang_err (np.array): angular error
    Returns:
        ang_err_stats (dict):  angular error statistics
    """
    ang_err = ang_err[~np.isnan(ang_err)]
    ang_err_stats = {"min": np.min(ang_err),
                     "10prc": np.percentile(ang_err, 10),
                     "median": np.median(ang_err),
                     "mean": np.mean(ang_err),
                     "90prc": np.percentile(ang_err, 90),
                     "max": np.max(ang_err)}
    return ang_err_stats